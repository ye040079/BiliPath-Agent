"""知识库单元测试：官方库共享 + 个人缓存隔离 + 收藏的用户隔离"""
from knowledge_base import KnowledgeBase


def test_personal_plan_save_and_find(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.db"))
    kb.save_plan("Python数据分析", "入门", 2.0, "", [{"bvid": "BV1"}], "个人计划内容",
                 total_weeks=8, user_id="alice")

    # 本人可命中
    found = kb.find_similar_plan("Python数据分析", level="入门", total_weeks=8, user_id="alice")
    assert found is not None
    assert found["topic"] == "Python数据分析"
    assert found["plan_content"] == "个人计划内容"
    assert found["videos"] == [{"bvid": "BV1"}]

    # 他人不可命中（历史/缓存隔离）
    assert kb.find_similar_plan("Python数据分析", level="入门", total_weeks=8, user_id="bob") is None

    # 个人历史列表只含本人
    assert len(kb.list_plans("alice")) == 1
    assert len(kb.list_plans("bob")) == 0


def test_official_plan_shared_and_protected(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.db"))
    kb.save_plan("线性代数", "入门", 2.0, "考研备考", [], "官方线性代数计划", total_weeks=8, is_official=True)
    # 用户生成同主题个人计划，不应覆盖官方行
    kb.save_plan("线性代数", "进阶", 3.0, "求职", [], "用户个人计划", total_weeks=6, user_id="u1")

    # 任何用户都命中官方（共享）
    hit = kb.find_similar_plan("线性代数", level="入门", total_weeks=8, user_id="anyone")
    assert hit is not None and hit.get("is_official") == 1
    assert hit["plan_content"] == "官方线性代数计划"

    # 官方行仍在、且能单独列出
    assert len(kb.list_official_plans()) == 1
    assert kb.get_plan(1, official=True)["plan_content"] == "官方线性代数计划"

    # 删除官方计划
    assert kb.delete_plan(1, official=True) is True


def test_plan_not_found(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.db"))
    assert kb.find_similar_plan("不存在的主题", user_id="u1") is None


def test_favorites_user_isolation(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.db"))
    assert kb.add_favorite("u1", "BV1", "t1", "a1", "url1") is True
    assert kb.add_favorite("u2", "BV2", "t2", "a2", "url2") is True

    favs_u1 = kb.list_favorites("u1")
    favs_u2 = kb.list_favorites("u2")
    assert len(favs_u1) == 1 and favs_u1[0]["bvid"] == "BV1"
    assert len(favs_u2) == 1 and favs_u2[0]["bvid"] == "BV2"

    # 同一用户重复收藏
    assert kb.add_favorite("u1", "BV1", "t1", "a1", "url1") is False

    # 删除仅影响目标用户
    assert kb.remove_favorite("u1", "BV1") is True
    assert kb.list_favorites("u1") == []
    assert len(kb.list_favorites("u2")) == 1
