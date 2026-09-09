"""
结构化输出模型（Pydantic）
所有 Agent 节点的 LLM 输出都通过这些模型做校验，
替代原来脆弱的正则 _parse_json + 多次重试。
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Intent(BaseModel):
    """协调者意图理解输出"""
    action: str = Field(default="execute", description="execute 或 clarify")
    topic: Optional[str] = None
    level: str = "入门"
    daily_hours: float = 2.0
    total_weeks: Optional[int] = None  # 学习周期总周数（用户选择/提到"X周"时），未指定为 None
    goal: str = ""
    question: Optional[str] = None
    confidence: float = 0.8


class SearchKeywords(BaseModel):
    """搜索关键词生成输出"""
    search_keywords: List[str] = Field(default_factory=list)


class VideoSummary(BaseModel):
    """视频内容总结输出"""
    bvid: str
    actual_content: str = ""
    key_points: List[str] = Field(default_factory=list)
    target_audience: str = ""
    quality: str = ""
    match_score: int = 5
    match_reason: str = ""
    confidence: str = "medium"


class Sentiment(BaseModel):
    """评论情感比例"""
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class CommentAnalysis(BaseModel):
    """评论分析输出"""
    bvid: str
    positive: List[str] = Field(default_factory=list)
    negative: List[str] = Field(default_factory=list)
    audience: str = ""
    questions: List[str] = Field(default_factory=list)
    sentiment: Sentiment = Field(default_factory=Sentiment)
    summary: str = ""


class ReviewItem(BaseModel):
    """审查后的推荐视频项"""
    rank: int = 0
    bvid: str = ""
    title: str = ""
    total_score: int = 0
    reason: str = ""
    warning: str = ""


class FilteredItem(BaseModel):
    """被过滤掉的视频及原因"""
    bvid: str = ""
    reason: str = ""


class ReviewResult(BaseModel):
    """审查 Agent 输出"""
    recommended: List[ReviewItem] = Field(default_factory=list)
    filtered_out: List[FilteredItem] = Field(default_factory=list)


class PlanAdjustment(BaseModel):
    """计划调整参数"""
    daily_hours: Optional[float] = None
    level: Optional[str] = None
    replace_video_index: int = -1
    # extend=延长整个学习周期(总周数) / shorten=缩短 / same=不变 —— 与每日时长无关
    duration_direction: str = "same"
    note: str = ""
