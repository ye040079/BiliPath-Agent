"""
用户认证模块：注册、登录、Token验证、密码加密
- 轻量级实现，不依赖第三方JWT库
- 密码用SHA256+盐加密
- Token用随机字符串，存在数据库
"""
import os
import json
import time
import hashlib
import secrets
import sqlite3
from typing import Optional, Dict
from loguru import logger


class AuthManager:
    """用户认证管理器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化用户表和Token表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                email TEXT DEFAULT '',
                custom_api_key TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_log(user_id)")
        conn.commit()
        conn.close()
        logger.info("[认证] 用户认证模块已初始化")

    # PBKDF2 迭代次数：OWASP 推荐对 SHA256 使用 60 万次量级
    _PBKDF2_ITERATIONS = 600_000

    def _hash_password(self, password: str, salt: str) -> str:
        """密码哈希：PBKDF2-HMAC-SHA256 + 每用户随机盐，抗彩虹表与暴力破解"""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            self._PBKDF2_ITERATIONS,
        ).hex()

    def register(self, username: str, password: str, email: str = "") -> Dict:
        """注册新用户"""
        # 校验
        if not username or len(username) < 2:
            return {"success": False, "error": "用户名至少2个字符"}
        if not password or len(password) < 6:
            return {"success": False, "error": "密码至少6个字符"}
        if len(username) > 32:
            return {"success": False, "error": "用户名最多32个字符"}

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 检查用户名是否存在
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        if c.fetchone():
            conn.close()
            return {"success": False, "error": "用户名已存在"}

        # 创建用户
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        try:
            c.execute(
                "INSERT INTO users (username, password_hash, salt, email) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, email)
            )
            user_id = c.lastrowid
            conn.commit()
            logger.info(f"[认证] 用户注册成功: {username} (id={user_id})")
            return {"success": True, "user_id": user_id, "username": username}
        except Exception as e:
            logger.error(f"[认证] 注册失败: {e}")
            return {"success": False, "error": f"注册失败: {str(e)}"}
        finally:
            conn.close()

    def login(self, username: str, password: str) -> Dict:
        """登录，返回Token"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, username, password_hash, salt FROM users WHERE username=?", (username,))
        row = c.fetchone()

        if not row:
            conn.close()
            return {"success": False, "error": "用户名或密码错误"}

        user_id, db_username, password_hash, salt = row
        # 验证密码
        input_hash = self._hash_password(password, salt)
        if input_hash != password_hash:
            conn.close()
            return {"success": False, "error": "用户名或密码错误"}

        # 生成Token（7天过期）
        token = secrets.token_hex(32)
        expires_at = time.time() + 7 * 24 * 3600
        c.execute(
            "INSERT INTO tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at)
        )
        c.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user_id,))
        conn.commit()
        conn.close()

        logger.info(f"[认证] 用户登录成功: {username}")
        return {
            "success": True,
            "token": token,
            "user_id": user_id,
            "username": db_username,
            "expires_at": expires_at,
        }

    def verify_token(self, token: str) -> Optional[Dict]:
        """验证Token，返回用户信息或None"""
        if not token:
            return None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT u.id, u.username, u.email, u.custom_api_key, t.expires_at
            FROM tokens t JOIN users u ON t.user_id = u.id
            WHERE t.token=?
        """, (token,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        # 检查过期
        if row["expires_at"] and float(row["expires_at"]) < time.time():
            logger.info(f"[认证] Token已过期: {token[:8]}...")
            return None

        return {
            "user_id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "custom_api_key": row["custom_api_key"] or "",
        }

    def logout(self, token: str):
        """登出，删除Token"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM tokens WHERE token=?", (token,))
        conn.commit()
        conn.close()

    def set_custom_api_key(self, user_id: int, api_key: str):
        """设置用户自定义API Key"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE users SET custom_api_key=? WHERE id=?", (api_key, user_id))
        conn.commit()
        conn.close()

    def get_usage_today(self, user_id: int) -> int:
        """获取用户今天的使用次数"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM usage_log
            WHERE user_id=? AND DATE(created_at)=DATE('now')
        """, (user_id,))
        count = c.fetchone()[0]
        conn.close()
        return count

    def log_usage(self, user_id: int, action: str = "chat"):
        """记录使用日志"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO usage_log (user_id, action) VALUES (?, ?)", (user_id, action))
        conn.commit()
        conn.close()

    def get_user_stats(self, user_id: int) -> Dict:
        """获取用户统计信息"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM usage_log WHERE user_id=?", (user_id,))
        total = c.fetchone()[0]
        c.execute("""
            SELECT COUNT(*) FROM usage_log
            WHERE user_id=? AND DATE(created_at)=DATE('now')
        """, (user_id,))
        today = c.fetchone()[0]
        conn.close()
        return {"total_uses": total, "today_uses": today}
