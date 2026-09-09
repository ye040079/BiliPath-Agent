"""
记忆系统：短期记忆（对话窗口）+ 长期记忆（用户偏好/历史）
- 短期记忆：当前对话的消息历史，滑动窗口，自动截断
- 长期记忆：用户偏好、学习历史、常用主题，跨会话持久化
"""
import json
import sqlite3
import os
from typing import List, Dict, Optional
from loguru import logger


class ShortTermMemory:
    """短期记忆：当前对话上下文，滑动窗口"""

    def __init__(self, max_tokens: int = 4000, max_messages: int = 20):
        self.messages: List[Dict] = []
        self.max_tokens = max_tokens
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        """添加一条消息"""
        self.messages.append({"role": role, "content": content})
        self._truncate()

    def add_system(self, content: str):
        """添加系统消息（不参与截断计数）"""
        self.messages.append({"role": "system", "content": content})

    def get_context(self) -> List[Dict]:
        """获取当前上下文"""
        return self.messages.copy()

    def get_recent(self, n: int = 5) -> List[Dict]:
        """获取最近n条消息"""
        return self.messages[-n:]

    def clear(self):
        """清空短期记忆"""
        self.messages = []

    def _truncate(self):
        """滑动窗口截断：保留最近的消息，控制token数"""
        # 先按消息数截断
        if len(self.messages) > self.max_messages:
            # 保留system消息
            system_msgs = [m for m in self.messages if m["role"] == "system"]
            other_msgs = [m for m in self.messages if m["role"] != "system"]
            self.messages = system_msgs + other_msgs[-(self.max_messages - len(system_msgs)):]

        # 再按token数粗略截断（按字符数估算）
        total_chars = sum(len(m["content"]) for m in self.messages)
        while total_chars > self.max_tokens * 3 and len(self.messages) > 3:
            # 删除最老的非system消息
            for i, m in enumerate(self.messages):
                if m["role"] != "system":
                    total_chars -= len(m["content"])
                    self.messages.pop(i)
                    break

    def __len__(self):
        return len(self.messages)


class LongTermMemory:
    """长期记忆：用户偏好、学习历史，跨会话持久化"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT DEFAULT '{}',
                learning_goals TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic TEXT,
                level TEXT,
                daily_hours REAL,
                plan_summary TEXT,
                video_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                summary TEXT,
                key_topics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON learning_history(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_summary_user ON conversation_summaries(user_id)")
        conn.commit()
        conn.close()

    def save_preference(self, user_id: str, key: str, value: str):
        """保存用户偏好"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT preferences FROM user_preferences WHERE user_id=?", (user_id,))
        row = c.fetchone()
        prefs = json.loads(row[0]) if row else {}
        prefs[key] = value
        c.execute("""
            INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (user_id, json.dumps(prefs, ensure_ascii=False)))
        conn.commit()
        conn.close()
        logger.info(f"[长期记忆] 用户{user_id}偏好已保存: {key}={value[:30]}")

    def get_preferences(self, user_id: str) -> Dict:
        """获取用户所有偏好"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT preferences FROM user_preferences WHERE user_id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        return json.loads(row[0]) if row else {}

    def record_learning(self, user_id: str, topic: str, level: str,
                        daily_hours: float, plan_summary: str, video_count: int):
        """记录一次学习规划"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO learning_history (user_id, topic, level, daily_hours, plan_summary, video_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, topic, level, daily_hours, plan_summary[:500], video_count))
        conn.commit()
        conn.close()
        logger.info(f"[长期记忆] 用户{user_id}学习记录已保存: {topic}")

    def get_learning_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户学习历史"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT * FROM learning_history
            WHERE user_id=? ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def save_conversation_summary(self, user_id: str, session_id: str,
                                   summary: str, key_topics: List[str]):
        """保存对话摘要（会话结束时）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversation_summaries (user_id, session_id, summary, key_topics)
            VALUES (?, ?, ?, ?)
        """, (user_id, session_id, summary[:1000], json.dumps(key_topics, ensure_ascii=False)))
        conn.commit()
        conn.close()
        logger.info(f"[长期记忆] 对话摘要已保存: session={session_id}")

    def get_relevant_memories(self, user_id: str, query: str, limit: int = 5) -> List[str]:
        """根据当前查询检索相关的长期记忆（简单关键词匹配）"""
        memories = []
        # 1. 用户偏好
        prefs = self.get_preferences(user_id)
        for k, v in prefs.items():
            if any(kw in query for kw in k.split()) or any(kw in str(v) for kw in query.split()[:3]):
                memories.append(f"用户偏好：{k}={v}")

        # 2. 学习历史
        history = self.get_learning_history(user_id, limit=10)
        query_words = set(query.lower().split())
        for h in history:
            topic_words = set(h["topic"].lower().split())
            if query_words & topic_words:
                memories.append(f"历史学习：{h['topic']}（{h['level']}，{h['daily_hours']}小时/天）")

        # 3. 对话摘要
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT summary, key_topics FROM conversation_summaries
            WHERE user_id=? ORDER BY created_at DESC LIMIT 10
        """, (user_id,))
        for row in c.fetchall():
            topics = json.loads(row[1]) if row[1] else []
            if any(t in query for t in topics):
                memories.append(f"历史对话：{row[0][:100]}")
        conn.close()

        return memories[:limit]

    def get_user_profile(self, user_id: str) -> str:
        """生成用户画像文本，注入到Agent系统提示中"""
        prefs = self.get_preferences(user_id)
        history = self.get_learning_history(user_id, limit=5)

        profile_parts = []
        if prefs:
            profile_parts.append("用户偏好：")
            for k, v in prefs.items():
                profile_parts.append(f"  - {k}: {v}")
        if history:
            profile_parts.append(f"\n近期学习过：{', '.join(h['topic'] for h in history[:3])}")

        return "\n".join(profile_parts) if profile_parts else "新用户，暂无历史记录"
