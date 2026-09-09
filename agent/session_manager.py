"""
会话管理器：支持多用户并发，每个session独立Agent实例
- 自动创建/回收会话
- 会话超时清理
- 线程安全
"""
import time
import uuid
import threading
from typing import Dict, Optional
from loguru import logger

from agent.graph import StudyAgent
from agent.memory import ShortTermMemory, LongTermMemory
from knowledge_base import KnowledgeBase


class Session:
    """单个用户会话"""

    def __init__(self, session_id: str, user_id: str, kb: KnowledgeBase, lt_memory: LongTermMemory):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.short_term = ShortTermMemory(max_tokens=4000, max_messages=20)
        self.agent = StudyAgent(knowledge_base=kb)
        self.agent.long_term_memory = lt_memory
        self.agent.user_id = user_id
        self.agent.session_id = session_id
        self.is_processing = False
        self.lock = threading.Lock()

    def touch(self):
        """更新最后活跃时间"""
        self.last_active = time.time()

    def is_expired(self, timeout: int = 3600) -> bool:
        """判断会话是否过期（默认1小时）"""
        return (time.time() - self.last_active) > timeout

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": len(self.short_term),
            "is_processing": self.is_processing,
        }


class SessionManager:
    """会话管理器：全局单例"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, kb: KnowledgeBase = None, lt_memory: LongTermMemory = None):
        if self._initialized:
            return
        self.sessions: Dict[str, Session] = {}
        self.kb = kb
        self.lt_memory = lt_memory
        self._cleanup_thread = None
        self._initialized = True
        self._start_cleanup()
        logger.info("[会话管理] 会话管理器已初始化")

    def _start_cleanup(self):
        """启动后台清理线程"""
        def cleanup_loop():
            while True:
                time.sleep(300)  # 每5分钟检查一次
                self._cleanup_expired()

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired(self):
        """清理过期会话"""
        expired = []
        for sid, session in self.sessions.items():
            if session.is_expired():
                expired.append(sid)

        for sid in expired:
            session = self.sessions.pop(sid, None)
            if session:
                logger.info(f"[会话管理] 会话已过期回收: {sid}")

    def get_or_create(self, session_id: str = None, user_id: str = "default") -> Session:
        """获取或创建会话"""
        with self._lock:
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                session.touch()
                return session

            # 创建新会话
            if not session_id:
                session_id = str(uuid.uuid4())[:8]

            session = Session(session_id, user_id, self.kb, self.lt_memory)
            self.sessions[session_id] = session
            logger.info(f"[会话管理] 新会话创建: {session_id} (用户: {user_id})")
            return session

    def get(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session

    def remove(self, session_id: str):
        """移除会话"""
        session = self.sessions.pop(session_id, None)
        if session:
            logger.info(f"[会话管理] 会话已移除: {session_id}")

    def stats(self) -> dict:
        """获取会话统计"""
        return {
            "active_sessions": len(self.sessions),
            "sessions": [s.to_dict() for s in self.sessions.values()],
        }

    def set_kb(self, kb: KnowledgeBase):
        """设置知识库（延迟初始化用）"""
        self.kb = kb

    def set_lt_memory(self, lt_memory: LongTermMemory):
        """设置长期记忆（延迟初始化用）"""
        self.lt_memory = lt_memory
