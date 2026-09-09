"""
项目配置文件
敏感信息（API Key）一律通过环境变量 / .env 文件注入，绝不硬编码在代码里。
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # 加载项目根目录 .env（强制 UTF-8，避免中文注释在 Windows GBK 下报错），不存在则忽略
    load_dotenv(Path(__file__).resolve().parent / ".env", encoding="utf-8")
except Exception:  # 未安装或读取失败时退化为纯环境变量
    pass

# ============ 大模型配置 ============
# 服务端默认 API Key：从环境变量 DEEPSEEK_API_KEY 读取。
# 部署时通过平台环境变量注入，本地开发写在 .env 里（见 .env.example）。
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 运行时配置（用户可在前端「设置」页覆盖，优先级高于默认）
# 注意：这里只是进程级运行时值，不存密钥；真实 Key 从上方环境变量读取。
LLM_API_KEY = ""
LLM_BASE_URL = DEFAULT_BASE_URL
LLM_MODEL = DEFAULT_MODEL
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# ============ 限流配置 ============
DAILY_LIMIT_PER_USER = 50       # 每个用户每天最多使用次数
ENABLE_RATE_LIMIT = True        # 是否启用限流

# ============ 注册开关 ============
# 公网演示时设为 false：先注册好自己的账号，再关闭注册，防止陌生人薅你的 API 配额
ALLOW_REGISTER = os.getenv("ALLOW_REGISTER", "true").lower() in ("1", "true", "yes")

# ============ 管理员（官方知识库管理） ============
# 填你注册的用户名；留空则任何人都不能管理官方知识库
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")

# ============ 共享库自动沉淀 ============
# 用户生成的新计划是否自动发布进共享(官方)库：默认 true，其他用户可复用；
# 若对外开放注册，建议改 false（只进个人历史），避免陌生人灌垃圾内容
AUTO_PUBLISH_OFFICIAL = os.getenv("AUTO_PUBLISH_OFFICIAL", "true").lower() in ("1", "true", "yes")

# ============ B站API配置 ============
BILIBILI_SEARCH_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
BILIBILI_VIDEO_INFO_URL = "https://api.bilibili.com/x/web-interface/view"
BILIBILI_COMMENT_URL = "https://api.bilibili.com/x/v2/reply/main"
BILIBILI_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"  # 获取 wbi 签名密钥
BILIBILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com",
}

# ============ Agent参数 ============
MAX_COMMENTS_PER_VIDEO = 80      # 每个视频爬取评论数
MAX_SEARCH_RESULTS = 15          # 搜索最大返回数
DEFAULT_DAILY_HOURS = 2.0        # 默认每日学习时长（小时）
AMBIGUITY_THRESHOLD = 0.7        # 歧义确认阈值
COARSE_FILTER_KEEP = 10          # 粗筛后保留候选数
FINAL_RECOMMEND_COUNT = 5        # 最终推荐视频数

# ============ 高级功能开关 ============
USE_REACT_MODE = False           # ReAct自主推理模式（True=Agent自主决策工具调用，False=协调者固定流程）
SESSION_TIMEOUT = 3600           # 会话超时时间（秒）
MAX_REACT_ITERATIONS = 8         # ReAct最大推理轮数
