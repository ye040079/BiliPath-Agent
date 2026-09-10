# 📚 BiliPath Agent · B站驱动的学习路线规划

> 基于 **LangGraph 编排 + 自研工具层**的 B站驱动学习路线规划 Agent：输入学习方向，自动搜索 B 站视频、审查视频实际内容、分析评论区真实反馈、生成个性化学习计划。支持共享学习方案复用、歧义确认、并行处理、多用户认证与流式进度。

## ✨ 核心特性

### LangGraph 编排层（图 = 流程）

| 节点 | 职责 |
|------|------|
| **understand_intent** | 意图理解 + 歧义检测（置信度 < 0.7 向用户确认） |
| **check_kb** | 知识库缓存检查（相似主题秒级复用） |
| **search** | 生成搜索关键词 → B 站搜索（wbi 签名）→ 粗筛去重 |
| **summarize** ∥ **analyze_comments** | 视频内容总结 与 评论分析 **并行执行** |
| **review** | 综合内容 + 评论，过滤不相关视频，加权排序 Top N |
| **generate_plan** | 生成三阶段学习计划 |

### 关键能力

- **🕸️ LangGraph 原生并行**：视频总结与评论分析是同一 superstep 的两个并行分支（async 节点 + `astream`），fan-out / fan-in
- **📡 原生流式进度**：`graph.astream(stream_mode="updates")` 逐节点产出进度，经 SSE 实时推送到前端
- **🧩 结构化输出**：LLM 输出用 Pydantic 模型校验 + 失败自动重试（把校验错误回喂给模型），替代脆弱的正则解析
- **🎬 视频内容审查**：不只信标题，通过字幕/简介/标签判断视频实际讲什么，过滤"养猪流"这类标题党
- **💬 评论区真实反馈**：提取好评、吐槽、适用人群、高频疑问
- **🧠 知识库缓存**：SQLite 缓存学习计划，相同/相似主题复用；命中需匹配 水平/每日时长/周期（选 6 周不会拿到 8 周旧计划）；搜索结果 24h 缓存
- **🎯 选择模式**：前端结构化表单直接选择 学习水平 / 学习周期(周) / 每日时长 / 学习目标，不依赖用户口述
- **📐 粒度自适应**：短周期（≤4 周）自动把计划细化到"每天学什么 / 看哪个视频 / 做什么练习"；长周期按周/阶段概括
- **❓ 歧义确认**：学习方向有歧义时主动向用户确认
- **🔐 多用户体系**：注册/登录（PBKDF2 密码哈希 + Bearer Token）、每日限流、会话隔离、收藏按用户隔离

## 🏗️ 架构

```
用户输入
  │
  ▼  understand_intent（意图理解 + 歧义检测）
  │     ├── 有歧义 → 向用户确认
  │     └── 明确 → 继续
  ▼  check_kb（知识库缓存）
  │     ├── 命中 → 直接返回缓存
  │     └── 未命中 → 继续
  ▼  search（搜索 Agent）
  │
  ├──────────────┬──────────────┐   ← 并行分支（fan-out）
  ▼              ▼
  summarize      analyze_comments   （视频总结 ∥ 评论分析）
  │              │
  └──────────────┴──────────────┘   ← 汇合（fan-in）
  ▼  review（审查 Agent，加权排序 + 过滤）
  ▼  generate_plan（三阶段学习计划）
  ▼  保存知识库 → 返回结果
```

**分层设计**：编排层交给 LangGraph（并行 / 流式 / 状态管理），工具层保持自研（B 站 API、SQLite 知识库、认证、记忆），这是业界标准的分层方式。

## 📁 项目结构

```
study-agent/
├── main.py                  # FastAPI 后端入口（REST + SSE 流式）
├── config.py                # 全局配置（密钥从环境变量读取）
├── knowledge_base.py        # SQLite 知识库（计划缓存 + 收藏 + 搜索缓存）
├── agent/
│   ├── graph.py             # LangGraph 编排层（节点 / 边 / 并行 / 流式）
│   ├── schemas.py           # Pydantic 结构化输出模型
│   ├── llm_client.py        # LLM 客户端（结构化输出 + 自动重试）
│   ├── prompts.py           # 各 Agent 系统提示词
│   ├── session_manager.py   # 会话管理（多用户并发）
│   ├── memory.py            # 短期记忆 + 长期记忆
│   └── auth.py              # 用户认证（PBKDF2 + Token）
├── tools/
│   └── bilibili_api.py      # B站 API（wbi 签名 + 字幕 + 评论）
├── static/
│   └── index.html           # 前端（原生 HTML/CSS/JS，响应式）
├── tests/                   # 单元测试
├── Dockerfile               # Docker 部署
├── requirements.txt
└── .env.example             # 环境变量模板（密钥不入库）
```

## 🚀 快速开始

### 1. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行

```bash
python main.py
```

浏览器打开 `http://localhost:7860`，注册账号登录后即可使用。

### 4. 运行测试

```bash
pytest
```

## 🐳 Docker 部署

```bash
docker build -t study-agent .
docker run -e DEEPSEEK_API_KEY=sk-xxx -p 7860:7860 study-agent
```

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 编排 | **LangGraph** | 图/状态机编排，并行分支 + 原生流式 |
| 后端 | FastAPI + Uvicorn | REST API + SSE 流式进度 |
| 结构化输出 | Pydantic v2 | LLM 输出校验 + 自动重试 |
| LLM 客户端 | httpx（自研） | OpenAI 兼容协议，支持 DeepSeek / 通义 / OpenAI |
| 数据 | B站公开 API | 搜索（wbi 签名）+ 字幕 + 评论 |
| 存储 | SQLite | 知识库缓存 + 收藏 + 认证 + 记忆 |
| 认证 | PBKDF2 + Token | 密码加盐哈希、7 天 Token、每日限流 |
| 前端 | 原生 HTML/CSS/JS | 响应式 + 实时进度 + Markdown 渲染 |
| 部署 | Docker | 一键部署 |

## 🔧 配置说明

`config.py` 中可调整的参数（密钥通过环境变量 `DEEPSEEK_API_KEY` 注入）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FINAL_RECOMMEND_COUNT` | 5 | 最终推荐视频数 |
| `COARSE_FILTER_KEEP` | 10 | 粗筛后保留候选数 |
| `AMBIGUITY_THRESHOLD` | 0.7 | 歧义确认阈值 |
| `MAX_COMMENTS_PER_VIDEO` | 80 | 每个视频爬取评论数 |
| `DEFAULT_DAILY_HOURS` | 2.0 | 默认每日学习时长 |
| `DAILY_LIMIT_PER_USER` | 50 | 每用户每日限流 |
| `ALLOW_REGISTER` | true | 是否允许公开注册（演示时设 false，只允许已有账号登录） |



## 📄 License

MIT License
