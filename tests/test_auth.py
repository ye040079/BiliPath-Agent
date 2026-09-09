"""认证模块单元测试：注册 / 登录 / Token 校验 / 密码哈希"""
import hashlib
import sqlite3

from agent.auth import AuthManager


def test_register_login_logout(tmp_path):
    am = AuthManager(str(tmp_path / "auth.db"))

    r = am.register("alice", "secret123", "alice@example.com")
    assert r["success"] is True

    # 重复用户名
    assert am.register("alice", "secret123")["success"] is False
    # 密码过短
    assert am.register("bob", "123")["success"] is False

    # 正确登录
    login = am.login("alice", "secret123")
    assert login["success"] is True
    token = login["token"]

    user = am.verify_token(token)
    assert user is not None
    assert user["username"] == "alice"

    # 错误密码
    assert am.login("alice", "wrongpass")["success"] is False

    # 登出后 token 失效
    am.logout(token)
    assert am.verify_token(token) is None


def test_password_hash_uses_pbkdf2(tmp_path):
    am = AuthManager(str(tmp_path / "auth.db"))
    am.register("bob", "secret123")

    conn = sqlite3.connect(str(tmp_path / "auth.db"))
    salt, password_hash = conn.execute(
        "SELECT salt, password_hash FROM users WHERE username='bob'"
    ).fetchone()
    conn.close()

    expected = hashlib.pbkdf2_hmac("sha256", b"secret123", salt.encode(), 600_000).hex()
    assert password_hash == expected
    # 不是明文、也不是简单 SHA256
    assert password_hash != "secret123"
