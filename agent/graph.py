"""
LangGraph 编排层：把原手写 Multi-Agent 协调器迁移为有向图。

- 节点：意图理解 → 知识库 → 搜索 → (视频总结 ∥ 评论分析) → 审查 → 计划生成
- 并行：视频总结与评论分析为同一 superstep 的并行分支（async 节点 + astream）
- 条件边：歧义确认、缓存命中直接结束
- 流式：graph.astream(stream_mode="updates") 逐节点产出进度

工具层（B站 API / SQLite 知识库 / 认证）保持自研，只把编排交给 LangGraph。
"""
import asyncio
import json
import re
from typing import TypedDict, List, Dict, Any, Optional

from loguru import logger

from langgraph.graph import StateGraph, START, END

import config
from agent.schemas import (
    Intent, SearchKeywords, VideoSummary, CommentAnalysis,
    ReviewResult, PlanAdjustment,
)
from agent.llm_client import LLMClient, LLMCallError, AgentInterrupted
from agent.prompts import (
    COORDINATOR_PROMPT, SEARCH_AGENT_PROMPT, VIDEO_SUMMARIZER_PROMPT,
    COMMENT_AGENT_PROMPT, REVIEWER_PROMPT, PLAN_GENERATOR_PROMPT, ADJUST_PROMPT,
)
from tools.bilibili_api import (
    search_videos, get_video_subtitles, get_comments, BilibiliAPIError,
)


class AgentState(TypedDict, total=False):
    """LangGraph 共享状态（图内逐节点流转）"""
    user_input: str
    # 意图
    intent: Dict[str, Any]
    # 流水线中间产物
    candidate_videos: List[Dict]
    video_summaries: List[Dict]
    comment_results: List[Dict]
    recommended: List[Dict]
    filtered_out: List[Dict]
    # 结果
    final_answer: str
    topic: str
    level: str
    daily_hours: float
    goal: str
    from_cache: bool
    error: Optional[str]


def _normalize_score(score) -> int:
    """把综合得分归一化到 0-100：LLM 有时按 0-10 给分（与 match_score 同量级），前端按 /100 渲染星"""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0
    if score <= 10:
        score *= 10
    return int(max(0, min(100, round(score))))


class StudyAgent:
    """基于 LangGraph 的学习规划 Agent（替代原手写协调器）"""

    # 节点名 → (进度阶段, 描述, 百分比)
    _STAGE_MAP = {
        "understand_intent": ("意图理解", "正在理解你的学习需求...", 10),
        "check_kb": ("知识库", "正在检查缓存...", 15),
        "search": ("搜索", "正在搜索B站视频...", 30),
        "summarize": ("内容审查", "正在总结视频实际内容...", 50),
        "analyze_comments": ("评论分析", "正在分析评论区反馈...", 50),
        "review": ("审查", "正在综合排序与过滤...", 80),
        "generate_plan": ("生成计划", "正在生成个性化学习计划...", 95),
    }

    def __init__(self, knowledge_base=None, llm: Optional[LLMClient] = None,
                 user_id: str = "default"):
        self.kb = knowledge_base
        self.llm = llm
        self.user_id = user_id
        self.session_id = None
        self.interrupted = False
        # 多轮对话状态
        self.conversation_history: List[Dict] = []
        self._pending_clarification: Optional[Dict] = None
        self.current_plan: Optional[Dict] = None
        # 从 config 读取阈值
        self.ambiguity_threshold = getattr(config, "AMBIGUITY_THRESHOLD", 0.7)
        self.coarse_filter_keep = getattr(config, "COARSE_FILTER_KEEP", 10)
        self.final_recommend_count = getattr(config, "FINAL_RECOMMEND_COUNT", 5)
        self.max_comments = getattr(config, "MAX_COMMENTS_PER_VIDEO", 80)
        # 用户生成的新计划是否自动发布进共享(官方)库：默认 true，其他人可复用；
        # 对外网开放注册时建议改 false（只进个人库），避免陌生人灌垃圾
        self.auto_publish_official = getattr(config, "AUTO_PUBLISH_OFFICIAL", True)
        self._graph = self._build_graph()

    def bind_llm(self, llm: LLMClient):
        """绑定当前请求的 LLM 客户端：注入 per-user api_key，并把中断标志接到 LLM 客户端"""
        self.llm = llm
        llm.interrupt_check = lambda: self.interrupted

    # ==================== 图构建 ====================

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("understand_intent", self._node_understand_intent)
        g.add_node("check_kb", self._node_check_kb)
        g.add_node("search", self._node_search)
        g.add_node("summarize", self._node_summarize)
        g.add_node("analyze_comments", self._node_analyze_comments)
        g.add_node("review", self._node_review)
        g.add_node("generate_plan", self._node_generate_plan)

        g.add_edge(START, "understand_intent")
        g.add_conditional_edges(
            "understand_intent", self._route_after_intent,
            {"check_kb": "check_kb", END: END},
        )
        g.add_conditional_edges(
            "check_kb", self._route_after_kb,
            {"search": "search", END: END},
        )
        # 并行 fan-out：搜索完成后，视频总结与评论分析并发执行
        g.add_edge("search", "summarize")
        g.add_edge("search", "analyze_comments")
        # fan-in：两者都完成后再进入审查
        g.add_edge(["summarize", "analyze_comments"], "review")
        g.add_edge("review", "generate_plan")
        g.add_edge("generate_plan", END)
        return g.compile()

    # ==================== 条件路由（同步，无 I/O） ====================

    def _route_after_intent(self, state: AgentState) -> str:
        intent = state.get("intent", {})
        return "check_kb" if intent.get("action") != "clarify" else END

    def _route_after_kb(self, state: AgentState) -> str:
        return END if state.get("from_cache") else "search"

    # ==================== 节点实现 ====================

    async def _node_understand_intent(self, state: AgentState) -> Dict[str, Any]:
        user_input = state["user_input"]
        if self._pending_clarification:
            original = self._pending_clarification["original"]
            question = self._pending_clarification["question"]
            self._pending_clarification = None
            merged = (
                f"【上下文】用户最初想学：{original}\n"
                f"【系统提问】{question}\n"
                f"【用户回答】{user_input}\n"
                f"请根据以上完整信息直接执行学习规划，不要再提问确认。"
            )
            intent = await self._parse_intent(merged, force_execute=True)
        else:
            intent = await self._parse_intent(user_input, force_execute=False)
        return {"intent": intent}

    async def _node_check_kb(self, state: AgentState) -> Dict[str, Any]:
        intent = state["intent"]
        topic = intent.get("topic") or state["user_input"]
        level = intent.get("level", "入门")
        daily_hours = float(intent.get("daily_hours", config.DEFAULT_DAILY_HOURS))
        goal = intent.get("goal", "")
        total_weeks = intent.get("total_weeks")

        if self.kb:
            cached = self.kb.find_similar_plan(
                topic, level=level, daily_hours=daily_hours, total_weeks=total_weeks,
                user_id=str(self.user_id),
            )
            if cached:
                logger.info(f"[知识库] 缓存命中: {cached['topic']}")
                self._bump("cache_hits")
                return {
                    "from_cache": True,
                    "cache_kind": "official" if cached.get("is_official") else "personal",
                    "final_answer": cached["plan_content"],
                    "topic": cached["topic"],
                    "level": cached.get("level", level),
                    "daily_hours": cached.get("daily_hours", daily_hours),
                    "total_weeks": cached.get("total_weeks"),
                    "recommended": cached.get("videos", []),
                }
        return {
            "from_cache": False,
            "topic": topic,
            "level": level,
            "daily_hours": daily_hours,
            "goal": goal,
            "total_weeks": intent.get("total_weeks"),
        }

    async def _node_search(self, state: AgentState) -> Dict[str, Any]:
        topic = state.get("topic") or state["user_input"]
        videos = await self._search_agent(topic)
        return {"candidate_videos": videos}

    async def _node_summarize(self, state: AgentState) -> Dict[str, Any]:
        videos = state.get("candidate_videos", [])
        topic = state.get("topic", "")
        level = state.get("level", "入门")
        summaries = await self._video_summarizer(videos, topic, level)
        return {"video_summaries": summaries}

    async def _node_analyze_comments(self, state: AgentState) -> Dict[str, Any]:
        videos = state.get("candidate_videos", [])
        topic = state.get("topic", "")
        candidates = self._quick_filter_for_comments(videos, topic)
        results = await self._comment_analyzer(candidates)
        return {"comment_results": results}

    async def _node_review(self, state: AgentState) -> Dict[str, Any]:
        topic = state.get("topic", "")
        level = state.get("level", "入门")
        result = await self._reviewer(
            topic, level,
            state.get("video_summaries", []),
            state.get("comment_results", []),
            state.get("candidate_videos", []),
        )
        if not result["recommended"]:
            raise RuntimeError(
                f"审查后没有符合要求的视频。搜索「{topic}」的视频可能与你的需求不匹配，请尝试调整关键词"
            )
        # 审查指标：候选(过滤+推荐)数与过滤数
        kept = result["recommended"]
        dropped = result.get("filtered_out", [])
        self._bump("review_considered", len(kept) + len(dropped))
        self._bump("review_filtered", len(dropped))
        return {
            "recommended": kept,
            "filtered_out": dropped,
        }

    async def _node_generate_plan(self, state: AgentState) -> Dict[str, Any]:
        topic = state.get("topic", "")
        level = state.get("level", "入门")
        daily_hours = float(state.get("daily_hours", config.DEFAULT_DAILY_HOURS))
        goal = state.get("goal", "")
        total_weeks = state.get("total_weeks") or state.get("intent", {}).get("total_weeks")
        recommended = state.get("recommended", [])
        final_answer = await self._generate_final_answer(
            topic, level, daily_hours, goal, recommended,
            state.get("comment_results", []),
            state.get("candidate_videos", []),
            {"filtered_out": state.get("filtered_out", [])},
            total_weeks=total_weeks,
        )
        # 保存到知识库
        if self.kb:
            try:
                self.kb.save_plan(
                    topic=topic, level=level, daily_hours=daily_hours, goal=goal,
                    videos=recommended, plan_content=final_answer,
                    total_weeks=total_weeks, user_id=str(self.user_id),
                    is_official=self.auto_publish_official,
                )
                logger.info(f"[知识库] 计划已保存: {topic}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[知识库] 保存失败: {e}")
        self._bump("plans_generated")
        return {
            "final_answer": final_answer,
            "topic": topic, "level": level, "daily_hours": daily_hours,
        }

    # ==================== 主入口 ====================

    async def arun(self, user_input: str):
        """异步流式主入口：逐条 yield 事件（progress / clarify / result）"""
        self.interrupted = False
        self.conversation_history.append({"role": "user", "content": user_input})
        logger.info(f"[协调者] 用户输入: {user_input}")

        # 优先级：计划调整（有当前计划且输入是调整指令）
        if self.current_plan and self._is_plan_adjustment(user_input):
            yield self._progress_event("generate_plan")
            result = await self._adjust_plan(user_input)
            if result.get("type") == "result":
                self.current_plan = result
            yield result
            return

        # 主流程：运行 LangGraph（astream 原生逐节点流式）
        initial: AgentState = {"user_input": user_input}
        final: Dict[str, Any] = dict(initial)
        async for chunk in self._graph.astream(initial, stream_mode="updates"):
            for node_name, update in chunk.items():
                if update:
                    final.update(update)
                evt = self._progress_event(node_name)
                if evt:
                    yield evt

        yield self._build_result(user_input, final)

    def _build_result(self, user_input: str, final: Dict[str, Any]) -> Dict[str, Any]:
        intent = final.get("intent", {})
        # 歧义确认
        if intent.get("action") == "clarify":
            question = intent.get("question") or "请告诉我你具体想学什么方向？"
            self._pending_clarification = {"original": user_input, "question": question}
            self.conversation_history.append({"role": "assistant", "content": f"确认问题：{question}"})
            return {
                "type": "clarify",
                "question": question,
                "confidence": intent.get("confidence", 0),
            }
        # 缓存命中
        if final.get("from_cache"):
            result = {
                "type": "result",
                "answer": final["final_answer"],
                "topic": final.get("topic"),
                "level": final.get("level"),
                "daily_hours": final.get("daily_hours"),
                "total_weeks": final.get("total_weeks"),
                "recommended_videos": final.get("recommended", []),
                "from_cache": True,
                "cache_kind": final.get("cache_kind", "personal"),
            }
            self.current_plan = result
            return result
        # 正常结果
        result = {
            "type": "result",
            "answer": final["final_answer"],
            "topic": final.get("topic"),
            "level": final.get("level"),
            "daily_hours": final.get("daily_hours"),
            "total_weeks": final.get("total_weeks"),
            "recommended_videos": final.get("recommended", []),
            "video_summaries": final.get("video_summaries", []),
            "comment_analysis": final.get("comment_results", []),
            "filtered_out": final.get("filtered_out", []),
            "candidate_count": len(final.get("candidate_videos", [])),
            "from_cache": False,
        }
        self.current_plan = result
        self.conversation_history.append({"role": "assistant", "content": final["final_answer"][:500]})
        return result

    def _progress_event(self, node_name: str) -> Optional[Dict[str, Any]]:
        """节点完成 → 进度事件（同时检查中断标志）"""
        if node_name not in self._STAGE_MAP:
            return None
        if self.interrupted:
            self.interrupted = False
            raise AgentInterrupted("用户中断了当前任务")
        stage, message, percent = self._STAGE_MAP[node_name]
        logger.info(f"[进度] {stage}: {message}")
        return {"type": "progress", "stage": stage, "message": message, "percent": percent}

    # ==================== 意图理解 ====================

    async def _parse_intent(self, text: str, force_execute: bool = False) -> Dict[str, Any]:
        history_text = ""
        if self.conversation_history:
            history_text = "\n【对话历史】\n" + "\n".join(
                f"{msg['role']}: {msg['content'][:100]}" for msg in self.conversation_history[-6:]
            )
        system = COORDINATOR_PROMPT
        if force_execute:
            system += "\n\n## 重要：这是用户回答确认问题后的继续执行，必须输出 action=execute，禁止 clarify。"
        user_content = f"{history_text}\n\n【当前输入】{text}"

        try:
            parsed = await self._call_structured(system, user_content, Intent, temperature=0.2)
            intent = parsed.model_dump()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[意图] 结构化解析失败，使用规则解析: {e}")
            intent = self._rule_parse_intent(text)

        if intent.get("action") == "clarify":
            return intent
        intent.setdefault("topic", text)
        intent.setdefault("level", "入门")
        intent.setdefault("daily_hours", config.DEFAULT_DAILY_HOURS)
        intent.setdefault("goal", "")
        # 兜底：LLM 漏提总周数时，用正则从文本补提（如"计划8周"）
        if not intent.get("total_weeks"):
            mw = re.search(r"(\d+)\s*周", text)
            if mw:
                intent["total_weeks"] = int(mw.group(1))
        intent["action"] = "execute"
        return intent

    def _rule_parse_intent(self, user_input: str) -> Dict[str, Any]:
        """LLM 解析失败时的规则兜底（只做解析，不造假数据）"""
        text = user_input.lower()
        daily_hours = config.DEFAULT_DAILY_HOURS
        m = re.search(r"每天?\s*(\d+(?:\.\d+)?)\s*[小个]?时?", user_input)
        if m:
            daily_hours = float(m.group(1))

        level = "入门"
        if any(w in text for w in ["进阶", "中级", "有基础"]):
            level = "进阶"
        elif any(w in text for w in ["高级", "精通", "深入"]):
            level = "高级"

        goal = ""
        if "考研" in text:
            goal = "考研备考"
        elif "找工作" in text or "求职" in text or "面试" in text:
            goal = "求职面试"

        if len(user_input) < 4 or user_input in ["学习", "我想学", "帮我"]:
            return {"action": "clarify", "question": "请告诉我你具体想学什么方向？", "confidence": 0.3}

        total_weeks = None
        mw = re.search(r"(\d+)\s*周", user_input)
        if mw:
            total_weeks = int(mw.group(1))

        return {
            "action": "execute", "topic": user_input, "level": level,
            "daily_hours": daily_hours, "goal": goal,
            "total_weeks": total_weeks, "confidence": 0.6,
        }

    # ==================== 搜索 Agent ====================

    async def _search_agent(self, topic: str) -> List[Dict]:
        keywords = [topic]
        try:
            parsed = await self._call_structured(
                SEARCH_AGENT_PROMPT,
                f"用户学习方向：{topic}\n请生成搜索关键词。",
                SearchKeywords, temperature=0.2,
            )
            if parsed.search_keywords:
                keywords = parsed.search_keywords
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[搜索] 关键词生成失败，用原主题: {e}")

        all_videos: List[Dict] = []
        for kw in keywords[:2]:
            if self.kb:
                cached = self.kb.get_search_cache(kw)
                if cached:
                    all_videos.extend(cached)
                    continue
            try:
                videos = await asyncio.to_thread(search_videos, kw, 20)
                all_videos.extend(videos)
                if self.kb:
                    self.kb.cache_search(kw, videos)
            except BilibiliAPIError as e:
                logger.warning(f"[搜索] 关键词 '{kw}' 搜索失败: {e}")

        if not all_videos:
            raise BilibiliAPIError(f"所有关键词搜索均失败: {keywords}")

        # 粗筛：去重 + 时长过滤
        seen, filtered = set(), []
        for v in all_videos:
            if v["bvid"] in seen:
                continue
            seen.add(v["bvid"])
            parts = v["duration"].split(":")
            total_min = 0
            try:
                if len(parts) == 3:
                    total_min = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 2:
                    total_min = int(parts[0])
            except ValueError:
                total_min = 0
            if total_min < 3:
                continue
            filtered.append(v)

        def _play_to_int(p):
            if isinstance(p, str):
                if "万" in p:
                    return float(p.replace("万", "")) * 10000
                try:
                    return float(p)
                except ValueError:
                    return 0
            return p or 0

        filtered.sort(key=lambda x: _play_to_int(x.get("play", 0)), reverse=True)
        return filtered[:self.coarse_filter_keep]

    # ==================== 视频总结 Agent ====================

    async def _video_summarizer(self, videos: List[Dict], topic: str, level: str) -> List[Dict]:
        video_data = []
        for v in videos:
            entry = {
                "bvid": v["bvid"], "title": v["title"],
                "description": v.get("description", ""), "tag": v.get("tag", ""),
                "typename": v.get("typename", ""), "author": v["author"],
                "duration": v["duration"],
            }
            try:
                subtitles = await asyncio.to_thread(get_video_subtitles, v["bvid"])
                entry["subtitles"] = (subtitles or "")[:3000]
                entry["has_subtitles"] = bool(subtitles)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[总结] 获取字幕失败 {v['bvid']}: {e}")
                entry["subtitles"] = ""
                entry["has_subtitles"] = False
            video_data.append(entry)

        all_summaries: List[Dict] = []
        batch_size = 3
        for i in range(0, len(video_data), batch_size):
            batch = video_data[i:i + batch_size]
            user_content = (
                f"用户需求：学习「{topic}」，水平：{level}\n\n"
                f"候选视频信息：\n{json.dumps(batch, ensure_ascii=False)}\n\n"
                f"总结每个视频实际内容并判断匹配度。"
            )
            try:
                summaries = await self._call_structured(
                    VIDEO_SUMMARIZER_PROMPT, user_content, list[VideoSummary], temperature=0.2,
                )
                all_summaries.extend([s.model_dump() for s in summaries])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[总结] 第{i // batch_size + 1}批解析失败: {e}")
                for v in batch:
                    all_summaries.append({
                        "bvid": v["bvid"], "actual_content": v["title"], "key_points": [],
                        "target_audience": "未知", "quality": "未知", "match_score": 5,
                        "match_reason": "基于标题推断", "confidence": "low",
                    })
        return all_summaries

    # ==================== 评论分析 Agent ====================

    def _quick_filter_for_comments(self, videos: List[Dict], topic: str) -> List[Dict]:
        topic_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", topic.lower()))
        scored = []
        for v in videos:
            title_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", v["title"].lower()))
            scored.append((len(topic_words & title_words), v))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in scored[:5]]

    async def _comment_analyzer(self, videos: List[Dict]) -> List[Dict]:
        comments_data = []
        fetch_count = min(self.max_comments, 40)
        for v in videos:
            try:
                comments = await asyncio.to_thread(get_comments, v["bvid"], fetch_count)
                comments_sorted = sorted(comments, key=lambda x: x.get("like", 0), reverse=True)[:20]
                comments_text = "\n".join(
                    f"[{c.get('like', 0)}赞] {c['content'][:80]}" for c in comments_sorted
                )
                comments_data.append({
                    "bvid": v["bvid"], "title": v.get("title", "")[:60],
                    "comments": comments_text[:2000], "total_count": len(comments),
                })
            except BilibiliAPIError as e:
                logger.warning(f"[评论] {v['bvid']} 评论获取失败: {e}")
                comments_data.append({
                    "bvid": v["bvid"], "title": v.get("title", ""),
                    "comments": "无评论数据", "total_count": 0,
                })

        if not comments_data:
            return []

        all_results: List[Dict] = []
        batch_size = 2
        for i in range(0, len(comments_data), batch_size):
            batch = comments_data[i:i + batch_size]
            user_content = f"分析评论区：\n{json.dumps(batch, ensure_ascii=False)}"
            try:
                results = await self._call_structured(
                    COMMENT_AGENT_PROMPT, user_content, list[CommentAnalysis], temperature=0.2,
                )
                all_results.extend([r.model_dump() for r in results])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[评论] 第{i // batch_size + 1}批解析失败: {e}")
                for v in batch:
                    all_results.append({
                        "bvid": v["bvid"], "positive": [], "negative": [], "audience": "未知",
                        "questions": [], "sentiment": {"positive": 0, "neutral": 0, "negative": 0},
                        "summary": "评论分析暂不可用",
                    })
        return all_results

    # ==================== 审查 Agent ====================

    async def _reviewer(self, topic: str, level: str, summaries: List[Dict],
                        comments: List[Dict], raw_videos: List[Dict]) -> Dict[str, Any]:
        sorted_by_score = sorted(summaries, key=lambda x: x.get("match_score", 0), reverse=True)
        review_data = []
        for s in sorted_by_score[:6]:
            bvid = s["bvid"]
            raw = next((v for v in raw_videos if v["bvid"] == bvid), {})
            c = next((v for v in comments if v["bvid"] == bvid), {})
            comment_summary = ""
            if c:
                pos = c.get("positive", [])
                neg = c.get("negative", [])
                if pos:
                    comment_summary = f"好评：{'; '.join(pos[:2])}"
                if neg:
                    comment_summary += f" 差评：{'; '.join(neg[:2])}"
            review_data.append({
                "bvid": bvid,
                "title": raw.get("title", s.get("actual_content", ""))[:60],
                "match_score": s.get("match_score", 5),
                "match_reason": s.get("match_reason", "")[:80],
                "comment_summary": comment_summary[:200],
            })

        top_n = self.final_recommend_count
        reviewer_prompt = REVIEWER_PROMPT.replace("{TOP_N}", str(top_n))
        user_content = (
            f"用户想学「{topic}」（{level}）。\n"
            f"候选视频：{json.dumps(review_data, ensure_ascii=False)}\n"
            f"过滤不相关的，选出 Top {top_n}。"
        )
        try:
            parsed = await self._call_structured(reviewer_prompt, user_content, ReviewResult, temperature=0.1)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[审查] 结构化解析失败，按匹配度排序兜底: {e}")
            parsed = None

        if not parsed or not parsed.recommended:
            return self._fallback_review(summaries, comments, raw_videos, top_n)

        result = parsed.model_dump()
        for r in result["recommended"]:
            bvid = r.get("bvid", "")
            raw = next((v for v in raw_videos if v["bvid"] == bvid), {})
            s = next((v for v in summaries if v["bvid"] == bvid), {})
            c = next((v for v in comments if v["bvid"] == bvid), {})
            r["url"] = raw.get("url", f"https://www.bilibili.com/video/{bvid}")
            r["author"] = raw.get("author", "未知")
            r["duration"] = raw.get("duration", "未知")
            r["play"] = raw.get("play", 0)
            r["total_score"] = _normalize_score(r.get("total_score", 0))
            if not r.get("title"):
                r["title"] = raw.get("title", s.get("actual_content", "未知标题"))
            if not r.get("reason") or len(r.get("reason", "").strip()) < 5:
                parts = []
                if s.get("match_reason"):
                    parts.append(s["match_reason"][:60])
                if c and c.get("positive"):
                    parts.append(f"评论区反馈：{c['positive'][0][:40]}")
                r["reason"] = "；".join(parts) if parts else "内容匹配度高，适合学习"
            r["summary"] = s
            r["comment"] = c
        return result

    def _fallback_review(self, summaries, comments, raw_videos, top_n):
        """审查解析失败时按匹配度排序兜底（使用已有数据，不造假）"""
        sorted_videos = sorted(summaries, key=lambda x: x.get("match_score", 0), reverse=True)
        recommended = []
        for i, s in enumerate(sorted_videos[:top_n]):
            raw = next((v for v in raw_videos if v["bvid"] == s["bvid"]), {})
            recommended.append({
                "rank": i + 1, "bvid": s["bvid"],
                "title": raw.get("title", s.get("actual_content", "未知标题")),
                "author": raw.get("author", "未知"), "duration": raw.get("duration", "未知"),
                "play": raw.get("play", 0), "total_score": s.get("match_score", 5) * 10,
                "reason": s.get("match_reason", "基于内容匹配度排序"), "warning": "",
                "url": raw.get("url", f"https://www.bilibili.com/video/{s['bvid']}"),
                "summary": s, "comment": next((c for c in comments if c["bvid"] == s["bvid"]), {}),
            })
        return {"recommended": recommended, "filtered_out": []}

    # ==================== 计划生成 ====================

    async def _generate_final_answer(self, topic, level, daily_hours, goal,
                                     recommended, comments, raw_videos, review_result,
                                     extra_instruction: str = "",
                                     total_weeks: Optional[int] = None) -> str:
        plan_videos = []
        for r in recommended:
            c = r.get("comment", {})
            plan_videos.append({
                "rank": r.get("rank"), "title": r.get("title", ""),
                "author": r.get("author", ""), "duration": r.get("duration", ""),
                "reason": r.get("reason", "")[:150], "warning": r.get("warning", "")[:100],
                "positive": c.get("positive", [])[:3], "negative": c.get("negative", [])[:3],
                "key_points": r.get("summary", {}).get("key_points", [])[:5],
            })
        plan_input = {
            "topic": topic, "level": level, "daily_hours": daily_hours,
            "goal": goal, "videos": plan_videos,
        }
        if total_weeks:
            plan_input["total_weeks"] = total_weeks
        plan_text = json.dumps(plan_input, ensure_ascii=False)

        user_content = f"根据以下信息生成学习计划：\n{plan_text}"
        max_gen_tokens = 1500
        if total_weeks:
            if total_weeks <= 4:
                # 短周期：细化到每天，让人能照着一天天执行
                max_gen_tokens = 2400
                user_content += (
                    f"\n\n用户明确选择的学习周期为 {total_weeks} 周（较短）。"
                    f"请【细化到每天】生成：先分成 2 个阶段覆盖这 {total_weeks} 周，"
                    f"再把每周内容拆成【每天】的具体安排——每天学什么知识点、看哪个视频、"
                    f"完成什么练习或检验（内容量按每天约 {daily_hours} 小时规划）。"
                    f"可以比平时细，但不要重复废话。"
                )
            elif total_weeks <= 9:
                user_content += (
                    f"\n\n用户明确选择的学习周期为 {total_weeks} 周："
                    f"按周规划，各阶段总周数必须正好覆盖 {total_weeks} 周，写明每阶段周数区间。"
                )
            else:
                # 长周期：按周概括，避免海量细节
                user_content += (
                    f"\n\n用户明确选择的学习周期为 {total_weeks} 周（较长）："
                    f"可以拆更多阶段，按周列出每周核心内容与对应视频即可，不要逐天排布。"
                )
        if extra_instruction:
            user_content += f"\n\n{extra_instruction}"

        response = ""
        for attempt in range(3):
            try:
                response = await self._call_text(
                    PLAN_GENERATOR_PROMPT,
                    user_content,
                    temperature=0.4 if attempt == 0 else 0.6,
                    max_tokens=max_gen_tokens, timeout=90,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[计划] 第{attempt + 1}次生成失败: {e}")
            if response and len(response.strip()) >= 20:
                break

        if not response or len(response.strip()) < 20:
            raise LLMCallError(
                "学习计划生成失败：LLM 多次调用均返回空内容。请检查 API Key 余额或稍后重试。"
            )

        header = f"## 📚 「{topic}」学习方案\n\n"
        header += f"**水平**：{level} | **每日学习**：{daily_hours}小时"
        if total_weeks:
            header += f" | **学习周期**：{total_weeks}周"
        if goal:
            header += f" | **目标**：{goal}"
        header += "\n\n---\n\n### 🎬 推荐视频（已通过内容审查）\n\n"

        for r in recommended:
            header += f"**{r['rank']}. [{r['title']}]({r.get('url', '')})**\n"
            header += f"   - UP主：{r.get('author', '未知')} | 时长：{r.get('duration', '未知')} | 综合得分：{_normalize_score(r.get('total_score', 0))}/100\n"
            if r.get("reason"):
                header += f"   - 推荐理由：{r['reason']}\n"
            if r.get("warning"):
                header += f"   - ⚠️ 注意：{r['warning']}\n"
            comment = next((c for c in comments if c.get("bvid") == r.get("bvid")), None)
            if comment:
                if comment.get("positive"):
                    header += f"   - 👍 好评：{'、'.join(comment['positive'][:3])}\n"
                if comment.get("negative"):
                    header += f"   - 👎 吐槽：{'、'.join(comment['negative'][:3])}\n"
            header += "\n"

        if review_result.get("filtered_out"):
            header += f"*已过滤 {len(review_result['filtered_out'])} 个不相关视频*\n\n"
        header += "---\n\n### 📅 学习计划\n\n"
        return header + response

    # ==================== 计划调整 ====================

    def _is_plan_adjustment(self, user_input: str) -> bool:
        adjustment_keywords = [
            "改成", "改为", "调整", "修改", "换一个", "换掉", "不要这个",
            "缩短", "延长", "减少", "增加", "每天", "时间改", "时长",
            "重新", "再来", "换个", "不喜欢", "太难", "太简单",
            "这个视频", "第一个", "第二个", "第三个",
        ]
        new_request_keywords = ["我想学", "帮我规划", "推荐", "学习路线", "怎么学"]
        text = user_input.lower()
        has_adjust = any(kw in text for kw in adjustment_keywords)
        is_new = any(kw in text for kw in new_request_keywords)
        return has_adjust and not is_new

    async def _adjust_plan(self, user_input: str) -> Dict[str, Any]:
        plan = self.current_plan or {}
        topic = plan.get("topic", "")
        level = plan.get("level", "入门")
        daily_hours = plan.get("daily_hours", 2.0)
        videos = plan.get("recommended_videos", [])

        try:
            parsed = await self._call_structured(
                ADJUST_PROMPT,
                f"当前计划主题：{topic}，水平：{level}，每日{daily_hours}小时。\n"
                f"推荐视频：\n" + "\n".join(
                    f"{v.get('rank', i + 1)}. {v.get('title', '')}" for i, v in enumerate(videos)
                ) + f"\n\n用户调整要求：{user_input}",
                PlanAdjustment, temperature=0.2,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[调整] 解析失败: {e}")
            parsed = None

        if not parsed:
            return {"type": "result", "answer": "未能理解调整要求，请换个说法。", "topic": topic}

        new_level = parsed.level or level
        replace_idx = parsed.replace_video_index

        # —— 学习周期（延长/缩短）调整：调总周数，不是每日时长 ——
        duration_direction = (parsed.duration_direction or "same").lower()
        mentioned_hours = bool(re.search(r"每天|每日|小时|时长", user_input))

        # 决定新的每日时长：仅延长/缩短周期且用户没提"每天X小时"时，时长保持不变
        new_daily = parsed.daily_hours if parsed.daily_hours is not None else daily_hours
        if duration_direction in ("extend", "shorten") and not mentioned_hours:
            new_daily = daily_hours

        # 组装给 LLM 的周期调整指令 + 展示用 note（以实际改动为准，避免 LLM 瞎解释）
        cycle_instruction = ""
        note_parts = []
        if duration_direction in ("extend", "shorten"):
            direction_cn = "延长" if duration_direction == "extend" else "缩短"
            expand = duration_direction == "extend"
            cycle_instruction = (
                f"用户希望【{direction_cn}整个学习周期】：请把总学习周数"
                f"{'拉长、每个阶段安排得更从容' if expand else '压缩、加快节奏'}，"
                "保持每日学习时长不变、推荐视频不变。"
            )
            note_parts.append(f"学习周期已{direction_cn}（总周数{'更多' if expand else '更少'}）")
        if new_daily != daily_hours:
            note_parts.append(f"每日时长调整为{new_daily}小时")
        if not note_parts and parsed.note:
            note_parts.append(parsed.note)
        note = "；".join(note_parts) or parsed.note or "计划已调整"

        # 把原计划文本喂给 LLM，让它只调整周期、不重写
        if cycle_instruction and plan.get("answer"):
            marker = "### 📅 学习计划"
            if marker in plan["answer"]:
                orig_plan = plan["answer"].split(marker, 1)[1].strip()
                cycle_instruction += f"\n\n【原学习计划，请只调整它的学习周期结构，不要重写推荐视频】\n{orig_plan[:1200]}"

        new_videos = videos
        if 0 < replace_idx <= len(videos):
            try:
                candidates = await self._search_agent(topic)
                current_bvids = {v["bvid"] for v in videos}
                replacement = next((c for c in candidates if c["bvid"] not in current_bvids), None)
                if replacement:
                    rep_reason = (
                        f"UP主{replacement.get('author', '未知')}制作，"
                        f"播放量{replacement.get('play', '未知')}，"
                        f"时长{replacement.get('duration', '未知')}，与{topic}方向匹配，作为替换推荐"
                    )
                    new_videos = videos.copy()
                    new_videos[replace_idx - 1] = {
                        "rank": replace_idx, "bvid": replacement["bvid"],
                        "title": replacement["title"], "author": replacement.get("author", "未知"),
                        "duration": replacement.get("duration", "未知"),
                        "total_score": 72,
                        "reason": rep_reason,
                        "url": replacement.get("url", f"https://www.bilibili.com/video/{replacement['bvid']}"),
                        "warning": "替换视频，建议先试听一集确认风格",
                    }
                    note += f"（已替换第{replace_idx}个视频）"
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[调整] 替换视频失败: {e}")

        # 延长/缩短周期时不锁死周数（让 LLM 按指令自由伸缩）；其余调整保留原周期
        reg_weeks = plan.get("total_weeks")
        if duration_direction in ("extend", "shorten"):
            reg_weeks = None

        new_answer = await self._generate_final_answer(
            topic, new_level, new_daily, plan.get("goal", ""), new_videos,
            plan.get("comment_analysis", []), new_videos,
            {"filtered_out": plan.get("filtered_out", [])},
            extra_instruction=cycle_instruction, total_weeks=reg_weeks,
        )
        return {
            "type": "result",
            "answer": f"🔄 **计划已调整**：{note}\n\n---\n\n{new_answer}",
            "topic": topic, "level": new_level, "daily_hours": new_daily,
            "recommended_videos": new_videos, "adjusted": True, "adjust_note": note,
            "total_weeks": reg_weeks,
        }

    # ==================== LLM / I/O 异步封装 ====================

    def _bump(self, key: str, n: int = 1):
        """运行指标计数（埋点失败不影响主流程）"""
        if self.kb:
            try:
                self.kb.incr_metric(key, n)
            except Exception:  # noqa: BLE001
                pass

    async def _call_structured(self, system, user, schema, temperature=None, max_tokens=None):
        self._bump("llm_calls")
        return await asyncio.to_thread(
            self.llm.complete_structured, system, user, schema, temperature, max_tokens
        )

    async def _call_text(self, system, user, temperature=None, max_tokens=None, timeout=None):
        self._bump("llm_calls")
        return await asyncio.to_thread(
            self.llm.complete, system, user, temperature, max_tokens, timeout
        )
