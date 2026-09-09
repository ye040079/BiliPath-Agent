"""
知识库模块：SQLite缓存学习计划，相似主题复用，定时检测新视频
"""
import sqlite3
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger


class KnowledgeBase:
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                topic_hash TEXT NOT NULL,
                level TEXT,
                daily_hours REAL,
                total_weeks INTEGER,
                goal TEXT,
                videos TEXT,
                plan_content TEXT,
                resources TEXT,
                user_id TEXT DEFAULT '',
                is_official INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                last_check_at TEXT,
                use_count INTEGER DEFAULT 1
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                keyword_hash TEXT NOT NULL,
                results TEXT,
                created_at TEXT,
                expire_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                bvid TEXT NOT NULL,
                title TEXT,
                author TEXT,
                url TEXT,
                topic TEXT,
                added_at TEXT
            )
        """)
        # 迁移：老库的 plans 表补充 total_weeks 列（结构化周期缓存匹配）
        plan_cols = [r[1] for r in c.execute("PRAGMA table_info(plans)").fetchall()]
        if "total_weeks" not in plan_cols:
            c.execute("ALTER TABLE plans ADD COLUMN total_weeks INTEGER")
        # 迁移：plans 补 user_id(所属用户) 与 is_official(官方库标记)
        if "user_id" not in plan_cols:
            c.execute("ALTER TABLE plans ADD COLUMN user_id TEXT DEFAULT ''")
        if "is_official" not in plan_cols:
            c.execute("ALTER TABLE plans ADD COLUMN is_official INTEGER DEFAULT 0")
        c.execute("CREATE INDEX IF NOT EXISTS idx_plans_owner ON plans(user_id, is_official)")
        # 迁移：老库的 favorites 表补充 user_id 列（多用户隔离）
        fav_cols = [r[1] for r in c.execute("PRAGMA table_info(favorites)").fetchall()]
        if "user_id" not in fav_cols:
            c.execute("ALTER TABLE favorites ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
        c.execute("CREATE INDEX IF NOT EXISTS idx_fav_user ON favorites(user_id)")
        # 运行指标表（LLM调用/缓存命中/审查过滤等计数器）
        c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL DEFAULT 0
            )
        """)
        # 用户聊天历史：每次成功回答（生成/命中）都记录一条，历史记录页的数据源
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                topic TEXT,
                level TEXT,
                daily_hours REAL,
                total_weeks INTEGER,
                goal TEXT,
                source TEXT DEFAULT 'fresh',
                plan_content TEXT,
                created_at TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_hist_user ON chat_history(user_id)")
        conn.commit()
        conn.close()
        logger.info(f"[知识库] 数据库初始化完成: {self.db_path}")

    def _topic_hash(self, topic: str) -> str:
        """生成主题哈希（归一化后）"""
        normalized = re.sub(r'[\s，。、！？,.!?]', '', topic.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    @staticmethod
    def _row_matches(row, level=None, daily_hours=None, total_weeks=None) -> bool:
        """缓存行是否满足请求的水平/每日时长/总周数约束（约束为 None 表示不限定该项）"""
        if not isinstance(row, dict):
            row = dict(row)
        if level and row.get("level") and row["level"] != level:
            return False
        if total_weeks is not None:
            rw = row.get("total_weeks")
            if rw is not None and int(rw) != int(total_weeks):
                return False
        if daily_hours is not None:
            rh = row.get("daily_hours")
            if rh is not None and abs(float(rh) - float(daily_hours)) > 0.01:
                return False
        return True

    def find_similar_plan(self, topic: str, threshold: float = 0.6,
                          level: str = None, daily_hours: float = None,
                          total_weeks: int = None,
                          user_id: str = None) -> Optional[Dict]:
        """
        两级知识库查找：
        1) 官方库（is_official=1，全用户共享，后台人工维护）
        2) 个人缓存（is_official=0 且属于该 user_id，只有自己命中）
        只有当水平/每日时长/总周数与缓存一致时才复用，避免"选 6 周拿到 8 周旧计划"。
        """
        normalized = re.sub(r'[\s，。、！？,.!?]', '', topic.lower().strip())
        h = self._topic_hash(topic)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # 1) 官方库精确匹配（共享）
        c.execute("SELECT * FROM plans WHERE topic_hash=? AND is_official=1 "
                  "ORDER BY updated_at DESC LIMIT 1", (h,))
        row = c.fetchone()
        if row and self._row_matches(row, level, daily_hours, total_weeks):
            logger.info(f"[知识库] 官方库命中: {row['topic']}")
            conn.close()
            return self._row_to_dict(row)

        # 2) 个人缓存精确匹配（仅自己）
        if user_id:
            c.execute("SELECT * FROM plans WHERE topic_hash=? AND is_official=0 AND user_id=? "
                      "ORDER BY updated_at DESC LIMIT 1", (h, user_id))
            row = c.fetchone()
            if row and self._row_matches(row, level, daily_hours, total_weeks):
                logger.info(f"[知识库] 个人缓存命中: {row['topic']}")
                conn.close()
                return self._row_to_dict(row)

        # 模糊匹配：官方（共享）优先，其次个人（仅自己）
        c.execute("SELECT * FROM plans WHERE is_official=1 ORDER BY use_count DESC LIMIT 20")
        rows = list(c.fetchall())
        if user_id:
            c.execute("SELECT * FROM plans WHERE is_official=0 AND user_id=? "
                      "ORDER BY use_count DESC LIMIT 20", (user_id,))
            rows += list(c.fetchall())
        conn.close()

        for row in rows:
            if not self._row_matches(row, level, daily_hours, total_weeks):
                continue
            cached_topic = re.sub(r'[\s，。、！？,.!?]', '', row['topic'].lower().strip())
            if normalized in cached_topic or cached_topic in normalized:
                overlap = len(set(normalized) & set(cached_topic)) / max(len(set(normalized)), 1)
                if overlap >= threshold:
                    logger.info(f"[知识库] 模糊匹配到计划: {row['topic']} (重叠度={overlap:.2f})")
                    return self._row_to_dict(row)

        return None

    def save_plan(self, topic: str, level: str, daily_hours: float, goal: str,
                  videos: List[Dict], plan_content: str, resources: List[Dict] = None,
                  total_weeks: int = None, user_id: str = None,
                  is_official: bool = False):
        """
        保存学习计划：
        - is_official=True → 官方库（后台维护、全用户共享，owner=''）
        - is_official=False → 个人缓存（owner=user_id）
        自动生成只 upsert 自己名下/官方之外的行，绝不会覆盖官方计划。
        """
        # 官方/共享行：保留生成者归属（后台手工录入时 user_id 为空）；
        # 个人行：归属默认 'default'
        if is_official:
            owner = user_id or ""
        else:
            owner = user_id or "default"
        official = 1 if is_official else 0
        now = datetime.now().isoformat()
        h = self._topic_hash(topic)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(
            "SELECT id, use_count FROM plans WHERE topic_hash=? AND user_id=? AND is_official=?",
            (h, owner, official),
        )
        existing = c.fetchone()

        if existing:
            c.execute("""
                UPDATE plans SET level=?, daily_hours=?, total_weeks=?, goal=?, videos=?, plan_content=?,
                resources=?, updated_at=?, last_check_at=?, use_count=use_count+1
                WHERE id=?
            """, (level, daily_hours, total_weeks, goal,
                  json.dumps(videos, ensure_ascii=False),
                  plan_content, json.dumps(resources or [], ensure_ascii=False),
                  now, now, existing[0]))
            logger.info(f"[知识库] 更新计划: {topic} (官方={is_official}) (次数={existing[1]+1})")
        else:
            c.execute("""
                INSERT INTO plans (topic, topic_hash, level, daily_hours, total_weeks, goal, videos,
                plan_content, resources, user_id, is_official, created_at, updated_at, last_check_at, use_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (topic, h, level, daily_hours, total_weeks, goal,
                  json.dumps(videos, ensure_ascii=False), plan_content,
                  json.dumps(resources or [], ensure_ascii=False), owner, official,
                  now, now, now))
            logger.info(f"[知识库] 保存新计划: {topic} (官方={is_official})")

        conn.commit()
        conn.close()

    def record_history(self, user_id: str, topic: str, level: str = "入门",
                       daily_hours: float = 2.0, total_weeks: int = None, goal: str = "",
                       source: str = "fresh", plan_content: str = ""):
        """记录一条用户学习历史（生成=fresh / 个人命中=personal / 共享命中=official）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_history (user_id, topic, level, daily_hours, total_weeks, goal, source, plan_content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, topic, level, daily_hours, total_weeks, goal, source,
             plan_content[:4000], datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def list_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """列出某用户的学习历史（含生成与命中的回答记录）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT id, topic, level, daily_hours, total_weeks, source, created_at "
            "FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_history(self, history_id: int, user_id: str) -> Optional[Dict]:
        """获取一条学习历史详情"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM chat_history WHERE id=? AND user_id=?", (history_id, user_id))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_history(self, history_id: int, user_id: str) -> bool:
        """删除一条学习历史"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM chat_history WHERE id=? AND user_id=?", (history_id, user_id))
        affected = c.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def count_history(self, user_id: str) -> int:
        """统计某用户的学习历史条数"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM chat_history WHERE user_id=?", (user_id,))
        count = c.fetchone()[0]
        conn.close()
        return count

    def list_plans(self, user_id: str, limit: int = 50) -> List[Dict]:
        """列出某用户的个人历史计划（含其自动发布进共享库的行，不含他人/后台手工的共享行）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT id, topic, level, daily_hours, total_weeks, goal, created_at, updated_at, use_count "
            "FROM plans WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def list_official_plans(self, limit: int = 200) -> List[Dict]:
        """列出官方库计划（后台管理用，含正文便于编辑回填）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT id, topic, level, daily_hours, total_weeks, goal, plan_content, created_at, updated_at "
            "FROM plans WHERE is_official=1 ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def count_own_plans(self, user_id: str) -> int:
        """统计某用户个人历史计划数（含其发布进共享库的行）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM plans WHERE user_id=?", (user_id,))
        count = c.fetchone()[0]
        conn.close()
        return count

    def get_plan(self, plan_id: int, user_id: str = None, official: bool = False) -> Optional[Dict]:
        """获取单个计划详情（个人按 user_id 隔离；official=True 用于后台）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        if official:
            c.execute("SELECT * FROM plans WHERE id=? AND is_official=1", (plan_id,))
        elif user_id:
            c.execute("SELECT * FROM plans WHERE id=? AND user_id=?", (plan_id, user_id))
        else:
            c.execute("SELECT * FROM plans WHERE id=?", (plan_id,))
        row = c.fetchone()
        conn.close()
        return self._row_to_dict(row) if row else None

    def delete_plan(self, plan_id: int, user_id: str = None, official: bool = False) -> bool:
        """删除计划：个人(默认按用户隔离)或官方(后台)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if official:
            c.execute("DELETE FROM plans WHERE id=? AND is_official=1", (plan_id,))
        else:
            c.execute("DELETE FROM plans WHERE id=? AND user_id=?", (plan_id, user_id))
        affected = c.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def cache_search(self, keyword: str, results: List[Dict], ttl_hours: int = 24):
        """缓存搜索结果"""
        h = hashlib.md5(keyword.lower().strip().encode()).hexdigest()
        now = datetime.now()
        expire = (now + timedelta(hours=ttl_hours)).isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM search_cache WHERE keyword_hash=?", (h,))
        c.execute("""
            INSERT INTO search_cache (keyword, keyword_hash, results, created_at, expire_at)
            VALUES (?, ?, ?, ?, ?)
        """, (keyword, h, json.dumps(results, ensure_ascii=False), now.isoformat(), expire))
        conn.commit()
        conn.close()

    def get_search_cache(self, keyword: str) -> Optional[List[Dict]]:
        """获取缓存的搜索结果"""
        h = hashlib.md5(keyword.lower().strip().encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM search_cache WHERE keyword_hash=?", (h,))
        row = c.fetchone()
        conn.close()

        if row:
            if datetime.now().isoformat() < row['expire_at']:
                logger.info(f"[知识库] 搜索缓存命中: {keyword}")
                return json.loads(row['results'])
            else:
                logger.info(f"[知识库] 搜索缓存已过期: {keyword}")
        return None

    def check_for_updates(self, plan_id: int) -> Dict:
        """
        检查计划是否有新视频更新
        返回：{"has_update": bool, "new_videos": [...], "message": "..."}
        """
        plan = self.get_plan(plan_id)
        if not plan:
            return {"has_update": False, "new_videos": [], "message": "计划不存在"}

        last_check = plan.get('last_check_at', '')
        now = datetime.now()

        # 24小时内不重复检查
        if last_check:
            last_check_dt = datetime.fromisoformat(last_check)
            if (now - last_check_dt).total_seconds() < 86400:
                return {"has_update": False, "new_videos": [],
                        "message": f"上次检查于{last_check_dt.strftime('%m-%d %H:%M')}，24小时内无需重复检查"}

        # 更新检查时间
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE plans SET last_check_at=? WHERE id=?", (now.isoformat(), plan_id))
        conn.commit()
        conn.close()

        # 实际的新视频检测由调用方（Multi-Agent）执行
        return {"has_update": None, "new_videos": [], "message": "已触发更新检查", "plan": plan}

    def add_favorite(self, user_id: str, bvid: str, title: str, author: str, url: str, topic: str = ""):
        """收藏视频（按用户隔离）"""
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM favorites WHERE user_id=? AND bvid=?", (user_id, bvid))
        if c.fetchone():
            conn.close()
            return False
        c.execute("""
            INSERT INTO favorites (user_id, bvid, title, author, url, topic, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, bvid, title, author, url, topic, now))
        conn.commit()
        conn.close()
        return True

    def list_favorites(self, user_id: str) -> List[Dict]:
        """列出某用户的收藏视频"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM favorites WHERE user_id=? ORDER BY added_at DESC", (user_id,))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def remove_favorite(self, user_id: str, bvid: str) -> bool:
        """取消收藏（按用户隔离）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM favorites WHERE user_id=? AND bvid=?", (user_id, bvid))
        affected = c.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def incr_metric(self, key: str, n: int = 1):
        """运行指标计数（LLM调用 / 缓存命中 / 审查过滤等）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO metrics(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = value + ?",
            (key, n, n),
        )
        conn.commit()
        conn.close()

    def get_metric_map(self) -> Dict:
        """读取全部运行指标计数"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            rows = c.execute("SELECT key, value FROM metrics").fetchall()
        except Exception:  # noqa: BLE001 老库暂无该表
            rows = []
        conn.close()
        return {k: v for k, v in rows}

    def get_stats(self, user_id: str = None) -> Dict:
        """获取知识库统计 + 运行指标；plan_count 指该用户的个人计划数"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if user_id:
            c.execute("SELECT COUNT(*) FROM plans WHERE user_id=?", (user_id,))
        else:
            c.execute("SELECT COUNT(*) FROM plans")
        plan_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM favorites WHERE user_id=?", (user_id,)) if user_id else c.execute("SELECT COUNT(*) FROM favorites")
        fav_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM chat_history WHERE user_id=?", (user_id,)) if user_id else c.execute("SELECT COUNT(*) FROM chat_history")
        history_count = c.fetchone()[0]
        c.execute("SELECT SUM(use_count) FROM plans")
        total_uses = c.fetchone()[0] or 0
        conn.close()

        m = self.get_metric_map()
        chats = m.get("chats", 0)
        plans = m.get("plans_generated", 0)
        cache_hits = m.get("cache_hits", 0)
        llm_calls = m.get("llm_calls", 0)
        considered = m.get("review_considered", 0)
        filtered = m.get("review_filtered", 0)

        return {
            "plan_count": plan_count,
            "favorite_count": fav_count,
            "history_count": history_count,
            "total_uses": total_uses,
            "metrics": {
                "chats": chats,
                "plans_generated": plans,
                "cache_hits": cache_hits,
                "llm_calls": llm_calls,
                "review_considered": considered,
                "review_filtered": filtered,
                "cache_hit_rate": round(cache_hits / chats, 4) if chats else 0,
                "avg_llm_calls_per_plan": round(llm_calls / plans, 1) if plans else 0,
                "filter_rate": round(filtered / considered, 4) if considered else 0,
            },
        }

    def _row_to_dict(self, row) -> Dict:
        d = dict(row)
        if d.get('videos'):
            d['videos'] = json.loads(d['videos'])
        if d.get('resources'):
            d['resources'] = json.loads(d['resources'])
        return d
