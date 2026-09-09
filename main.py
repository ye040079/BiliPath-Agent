"""
BiliPath Agent - B站驱动的学习路线规划后端
运行方式：python main.py
访问：http://localhost:7860
"""
import os
import sys
import json
import asyncio
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from loguru import logger

import config
from agent.graph import StudyAgent
from agent.llm_client import LLMClient, LLMCallError, AgentInterrupted
from agent.session_manager import SessionManager
from agent.memory import LongTermMemory
from agent.auth import AuthManager
from tools.bilibili_api import BilibiliAPIError
from knowledge_base import KnowledgeBase

# ============ 初始化 ============
app = FastAPI(title="BiliPath Agent", version="1.0")

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.db")
kb = KnowledgeBase(DB_PATH)
lt_memory = LongTermMemory(DB_PATH)
auth = AuthManager(DB_PATH)
# 会话管理器：支持多用户并发
session_manager = SessionManager(kb=kb, lt_memory=lt_memory)
# 兼容旧代码的全局agent（单用户模式用）
agent = StudyAgent(knowledge_base=kb)

# 静态文件目录
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ============ 请求模型 ============
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ConfigRequest(BaseModel):
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


class FavoriteRequest(BaseModel):
    bvid: str
    title: str
    author: str = ""
    url: str = ""
    topic: str = ""


# ============ 认证请求模型 ============
class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class ApiKeyRequest(BaseModel):
    api_key: str = ""


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """从 Authorization: Bearer <token> 请求头提取 token"""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def get_optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """FastAPI 依赖：从请求头解析当前用户，未登录返回 None"""
    token = _extract_bearer_token(authorization)
    return auth.verify_token(token) if token else None


def require_user(user: Optional[dict] = Depends(get_optional_user)) -> dict:
    """登录鉴权依赖：未登录抛 401"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def resolve_api_key(user: dict = None) -> str:
    """解析API Key：用户自定义优先，否则用服务端默认"""
    if user and user.get("custom_api_key"):
        return user["custom_api_key"]
    return config.DEFAULT_API_KEY or config.LLM_API_KEY


# ============ 页面路由 ============
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>前端页面未找到，请确保 static/index.html 存在</h1>")


@app.get("/api/info")
async def app_info():
    """公开信息（无需登录），前端据此决定是否展示注册入口"""
    return {"allow_register": config.ALLOW_REGISTER, "app": "BiliPath Agent"}


# ============ API: 认证 ============
@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """用户注册（受 ALLOW_REGISTER 开关控制）"""
    if not config.ALLOW_REGISTER:
        raise HTTPException(status_code=403, detail="注册已关闭，请联系管理员")
    result = auth.register(req.username, req.password, req.email)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """用户登录，返回Token"""
    result = auth.login(req.username, req.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return result


@app.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(default=None)):
    """用户登出"""
    token = _extract_bearer_token(authorization)
    if token:
        auth.logout(token)
    return {"success": True}


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(require_user)):
    """获取当前用户信息"""
    stats = auth.get_user_stats(user["user_id"])
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "email": user["email"],
        "has_custom_api_key": bool(user["custom_api_key"]),
        "is_admin": bool(config.ADMIN_USERNAME) and user["username"] == config.ADMIN_USERNAME,
        "stats": stats,
        "daily_limit": config.DAILY_LIMIT_PER_USER,
    }


@app.post("/api/auth/api-key")
async def set_user_api_key(req: ApiKeyRequest, user: dict = Depends(require_user)):
    """设置当前用户的自定义 API Key"""
    api_key = req.api_key.strip()
    auth.set_custom_api_key(user["user_id"], api_key)
    return {"success": True, "message": "API Key已保存" if api_key else "API Key已清除"}


# ============ API: 对话 ============
@app.post("/api/chat")
async def chat(req: ChatRequest, user: dict = Depends(require_user)):
    """
    对话接口：SSE 流式返回进度和最终结果
    - 每个会话独立 StudyAgent 实例 + 独立 LangGraph 运行
    - per-user API Key 依赖注入，消除多用户并发竞态
    - LangGraph astream 原生逐节点流式
    """
    # 1. 限流检查
    if config.ENABLE_RATE_LIMIT:
        today_uses = auth.get_usage_today(user["user_id"])
        if today_uses >= config.DAILY_LIMIT_PER_USER:
            raise HTTPException(
                status_code=429,
                detail=f"今日使用次数已达上限（{config.DAILY_LIMIT_PER_USER}次），请明天再用"
            )

    # 2. 解析 API Key（用户自定义优先，否则服务端默认）
    api_key = resolve_api_key(user)
    if not api_key:
        raise HTTPException(status_code=400, detail="服务端未配置API Key，请联系管理员或在设置中填写自己的Key")

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 3. 获取或创建会话（多用户并发核心，按用户隔离）
    session = session_manager.get_or_create(req.session_id, user_id=str(user["user_id"]))
    session_agent = session.agent

    # 会话级锁：防止同一用户并发请求
    if not session.lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="当前会话正在处理中，请稍候")

    # 4. per-user LLM 客户端（依赖注入，不污染全局配置）
    llm_client = LLMClient(
        api_key=api_key,
        base_url=config.LLM_BASE_URL or config.DEFAULT_BASE_URL,
        model=config.LLM_MODEL or config.DEFAULT_MODEL,
    )
    session_agent.bind_llm(llm_client)

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        heartbeat_interval = 15  # 秒：无事件超时则发一条注释心跳，防止代理/ngrok 空闲掐断

        # 把 agent 产出收敛进队列：事件 / 各类异常 / 结束
        async def _agent_producer():
            try:
                async for ev in session_agent.arun(req.message):
                    await queue.put(("event", ev))
            except AgentInterrupted:
                await queue.put(("interrupted", None))
            except LLMCallError as e:
                await queue.put(("llm_error", str(e)))
            except BilibiliAPIError as e:
                await queue.put(("bili_error", str(e)))
            except Exception as e:  # noqa: BLE001
                logger.exception(f"[API] 对话出错: {e}")
                await queue.put(("error", f"{type(e).__name__}: {str(e)}"))
            finally:
                await queue.put(("done", None))

        def _sse(obj: dict) -> str:
            obj["session_id"] = session.session_id
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        try:
            session.is_processing = True
            auth.log_usage(user["user_id"], "chat")
            try:
                kb.incr_metric("chats")
            except Exception:  # noqa: BLE001
                pass
            session.short_term.add("user", req.message)

            # 首个进度事件：连接一建立立刻有数据，避免传输层空闲超时
            yield _sse({"type": "progress", "stage": "开始", "message": "请求已收到，Agent 正在工作...", "percent": 5})

            producer = asyncio.create_task(_agent_producer())
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
                except asyncio.TimeoutError:
                    # SSE 注释行心跳：前端按 data: 前缀过滤会自动忽略，仅用于保活防掐断
                    yield ": heartbeat\n\n"
                    continue

                if kind == "done":
                    break
                if kind == "event":
                    event = payload
                    if event.get("type") == "result":
                        # 大字段不返回前端，但保留在会话内供后续"计划调整"使用
                        event = _serialize_result(event)
                        session.short_term.add("assistant", event.get("answer", "")[:500])
                        # 记录到"我的历史"（生成与命中都记，来源区分）
                        try:
                            kb.record_history(
                                user_id=str(user["user_id"]),
                                topic=event.get("topic") or req.message,
                                level=event.get("level") or "入门",
                                daily_hours=event.get("daily_hours") or config.DEFAULT_DAILY_HOURS,
                                total_weeks=event.get("total_weeks"),
                                goal=event.get("goal") or "",
                                source=event.get("cache_kind") or "fresh",
                                plan_content=event.get("answer", ""),
                            )
                        except Exception as e:
                            logger.warning(f"[历史] 记录失败: {e}")
                        try:
                            lt_memory.record_learning(
                                session.user_id, event.get("topic", req.message),
                                event.get("level", "入门"), event.get("daily_hours", 2.0),
                                event.get("answer", "")[:300],
                                len(event.get("recommended_videos", []))
                            )
                        except Exception as e:
                            logger.warning(f"[长期记忆] 记录失败: {e}")
                    yield _sse(event)
                    continue

                # 异常 → 对应错误事件
                if kind == "llm_error":
                    yield _sse({"type": "error", "message": f"LLM调用失败：{payload}"})
                elif kind == "bili_error":
                    yield _sse({"type": "error", "message": f"B站数据获取失败：{payload}"})
                elif kind == "interrupted":
                    logger.info("[API] 任务已中断")
                    yield _sse({"type": "interrupted", "message": "已中断当前任务"})
                else:
                    yield _sse({"type": "error", "message": payload})
                break
        finally:
            try:
                producer.cancel()
            except Exception:  # noqa: BLE001
                pass
            session.is_processing = False
            session.lock.release()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _serialize_result(result: dict) -> dict:
    """序列化结果，移除不可JSON序列化的内容"""
    out = {}
    for k, v in result.items():
        if k in ("video_summaries", "comment_analysis", "filtered_out"):
            continue  # 这些太大，不返回给前端
        out[k] = v
    return out


# ============ API: 中断 ============
class InterruptRequest(BaseModel):
    session_id: Optional[str] = None

@app.post("/api/interrupt")
async def interrupt(req: InterruptRequest, user: dict = Depends(require_user)):
    """中断当前正在执行的任务（支持会话级）"""
    session_id = req.session_id
    if session_id:
        session = session_manager.get(session_id)
        if session:
            session.agent.interrupted = True
            logger.info(f"[API] 会话{session_id}中断请求已发送")
            return {"success": True, "message": "中断请求已发送"}
    # 兼容旧版：中断全局agent
    agent.interrupted = True
    logger.info("[API] 全局中断请求已发送")
    return {"success": True, "message": "中断请求已发送"}


# ============ API: 会话管理 ============
@app.get("/api/session/stats")
async def session_stats(user: dict = Depends(require_user)):
    """获取会话统计（监控用）"""
    return session_manager.stats()

@app.post("/api/session/clear")
async def clear_session(req: InterruptRequest, user: dict = Depends(require_user)):
    """清除指定会话"""
    if req.session_id:
        session_manager.remove(req.session_id)
        return {"success": True}
    return {"success": False, "message": "需要session_id"}


# ============ API: 配置 ============
@app.post("/api/config")
async def set_config(req: ConfigRequest, user: dict = Depends(require_user)):
    """保存当前用户的自定义 API Key 并测试连接（用独立客户端，不污染全局配置）"""
    api_key = req.api_key.strip()
    base_url = req.base_url.strip().rstrip("/")
    model = req.model.strip()

    try:
        test_client = LLMClient(api_key=api_key, base_url=base_url, model=model, timeout=30)
        test_client.complete("你是API测试助手", "请回复'连接成功'")
        ok, err = True, ""
    except Exception as e:
        ok, err = False, str(e)

    if ok:
        auth.set_custom_api_key(user["user_id"], api_key)
        return {"success": True, "message": "配置成功，API连接正常", "model": model}
    return {"success": False, "message": f"API连接失败：{err}", "model": model}


@app.get("/api/config")
async def get_config(user: dict = Depends(require_user)):
    """获取当前用户可用的配置状态"""
    return {
        "configured": bool(resolve_api_key(user)),
        "model": config.LLM_MODEL or config.DEFAULT_MODEL,
        "base_url": config.LLM_BASE_URL or config.DEFAULT_BASE_URL,
        "has_custom_key": bool(user.get("custom_api_key")),
    }


# ============ API: 历史记录（个人计划，需登录） ============
@app.get("/api/plans")
async def list_plans(user: dict = Depends(require_user)):
    """列出当前用户的学习历史（生成与缓存命中都会记录）"""
    return {"plans": kb.list_history(str(user["user_id"]))}


@app.get("/api/plans/{plan_id}")
async def get_plan(plan_id: int, user: dict = Depends(require_user)):
    """获取一条学习历史详情"""
    plan = kb.get_history(plan_id, str(user["user_id"]))
    if not plan:
        raise HTTPException(status_code=404, detail="记录不存在")
    return plan


@app.delete("/api/plans/{plan_id}")
async def delete_plan(plan_id: int, user: dict = Depends(require_user)):
    """删除一条学习历史"""
    if kb.delete_history(plan_id, str(user["user_id"])):
        return {"success": True}
    raise HTTPException(status_code=404, detail="记录不存在")


@app.post("/api/plans/{plan_id}/refresh")
async def refresh_plan(plan_id: int, user: dict = Depends(require_user)):
    """检查计划是否有新视频更新"""
    result = kb.check_for_updates(plan_id)
    return result


# ============ API: 官方知识库管理（仅管理员，前端不可见普通用户入口） ============
def require_admin(user: dict = Depends(require_user)) -> dict:
    """管理员鉴权：用户名需等于 config.ADMIN_USERNAME"""
    if not config.ADMIN_USERNAME or user.get("username") != config.ADMIN_USERNAME:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


class AdminPlanRequest(BaseModel):
    topic: str
    level: str = "入门"
    daily_hours: float = 2.0
    total_weeks: Optional[int] = None
    goal: str = ""
    plan_content: str = ""


@app.get("/api/admin/plans")
async def admin_list_plans(user: dict = Depends(require_admin)):
    """列出官方知识库计划"""
    return {"plans": kb.list_official_plans()}


@app.get("/api/admin/plans/{plan_id}")
async def admin_get_plan(plan_id: int, user: dict = Depends(require_admin)):
    """获取官方计划详情（编辑回填用）"""
    plan = kb.get_plan(plan_id, official=True)
    if not plan:
        raise HTTPException(status_code=404, detail="计划不存在")
    return plan


@app.post("/api/admin/plans")
async def admin_save_plan(req: AdminPlanRequest, user: dict = Depends(require_admin)):
    """新建/更新官方计划（按主题 upsert，自动缓存不会覆盖官方行）"""
    if not req.topic.strip() or not req.plan_content.strip():
        raise HTTPException(status_code=400, detail="主题和计划内容不能为空")
    kb.save_plan(
        topic=req.topic.strip(), level=req.level, daily_hours=req.daily_hours,
        goal=req.goal, total_weeks=req.total_weeks,
        videos=[], plan_content=req.plan_content, is_official=True,
    )
    return {"success": True}


@app.delete("/api/admin/plans/{plan_id}")
async def admin_delete_plan(plan_id: int, user: dict = Depends(require_admin)):
    """删除官方计划"""
    if kb.delete_plan(plan_id, official=True):
        return {"success": True}
    raise HTTPException(status_code=404, detail="计划不存在")


# ============ API: 收藏（按用户隔离） ============
@app.get("/api/favorites")
async def list_favorites(user: dict = Depends(require_user)):
    return {"favorites": kb.list_favorites(str(user["user_id"]))}


@app.post("/api/favorites")
async def add_favorite(req: FavoriteRequest, user: dict = Depends(require_user)):
    ok = kb.add_favorite(str(user["user_id"]), req.bvid, req.title, req.author, req.url, req.topic)
    return {"success": ok}


@app.delete("/api/favorites/{bvid}")
async def remove_favorite(bvid: str, user: dict = Depends(require_user)):
    ok = kb.remove_favorite(str(user["user_id"]), bvid)
    return {"success": ok}


# ============ API: 统计 ============
@app.get("/api/stats")
async def get_stats(user: dict = Depends(require_user)):
    """个人统计人人可见；运行指标(缓存命中/LLM调用/审查过滤)属后台数据，仅管理员返回"""
    stats = kb.get_stats(str(user["user_id"]))
    is_admin = bool(config.ADMIN_USERNAME) and user["username"] == config.ADMIN_USERNAME
    if not is_admin:
        stats.pop("total_uses", None)
        stats.pop("metrics", None)
    return stats


# ============ 启动 ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print("=" * 50)
    print("  BiliPath Agent v1.0")
    print(f"  访问地址: http://localhost:{port}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)
