"""
LLM 客户端（OpenAI 兼容协议）
- 基于 httpx，支持任意 OpenAI 兼容后端（DeepSeek / 通义千问 / OpenAI）
- complete(): 普通文本补全
- complete_structured(): JSON 指令 + Pydantic 校验 + 自动重试（把校验错误回喂给模型）
- 每次请求显式持有 api_key/base_url/model，消除全局可变配置带来的多用户并发竞态
"""
import json
import re
from typing import Optional, TypeVar, Callable, Any
import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError
from loguru import logger


class LLMCallError(Exception):
    """LLM 调用失败"""
    pass


class AgentInterrupted(Exception):
    """用户主动中断任务"""
    pass


T = TypeVar("T")


def _extract_json(text: str) -> Any:
    """从 LLM 输出中鲁棒提取 JSON（对象或数组均可）"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 或 ``` ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    # 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 找第一个 { 或 [ 并做括号配对提取
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return None


class LLMClient:
    """线程安全的轻量 LLM 客户端，每次调用显式传入配置，不依赖全局状态"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 60,
        interrupt_check: Optional[Callable[[], bool]] = None,
    ):
        if not api_key:
            raise LLMCallError("未配置 API Key，请通过环境变量 DEEPSEEK_API_KEY 或前端设置")
        self.api_key = api_key
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # 可选的中断检查回调（返回 True 表示应中断）
        self.interrupt_check = interrupt_check

    def _check_interrupt(self):
        if self.interrupt_check and self.interrupt_check():
            raise AgentInterrupted("用户中断了当前任务")

    def _post(self, payload: dict, timeout: int) -> dict:
        self._check_interrupt()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise LLMCallError(f"LLM 请求超时（{timeout}s）")
        except httpx.ConnectError:
            raise LLMCallError(f"无法连接 LLM API: {self.base_url}")
        except Exception as e:  # noqa: BLE001
            raise LLMCallError(f"LLM 请求异常: {str(e)[:200]}")

        if resp.status_code != 200:
            raise LLMCallError(f"LLM 返回 {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if not data.get("choices"):
            raise LLMCallError(f"LLM 返回格式异常: {str(data)[:200]}")
        return data

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """普通文本补全，返回字符串内容"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        data = self._post(payload, timeout or self.timeout)
        self._check_interrupt()
        return data["choices"][0]["message"]["content"]

    def complete_structured(
        self,
        system_prompt: str,
        user_content: str,
        schema: Any,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
    ) -> T:
        """
        JSON 指令 + Pydantic 校验 + 自动重试的结构化输出。
        schema 可以是单个 BaseModel 子类，也可以是 list[BaseModel]（用 TypeAdapter 校验）。
        失败时把校验错误回喂给模型，让模型自我修正。
        """
        adapter: TypeAdapter = TypeAdapter(schema)
        prompt = f"{user_content}\n\n请严格只输出一个 JSON（对象或数组），不要任何解释、不要 markdown 代码块。"
        last_err = ""
        for _ in range(max_retries + 1):
            content = self.complete(
                system_prompt,
                prompt + (f"\n\n上一次输出因校验失败被拒绝：{last_err}\n请修正后重新输出。" if last_err else ""),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            obj = _extract_json(content)
            if obj is None:
                last_err = "输出不是合法 JSON"
                continue
            try:
                return adapter.validate_python(obj)
            except ValidationError as e:
                last_err = str(e)[:500]
                continue
        raise LLMCallError(f"结构化输出解析失败（重试 {max_retries} 次后仍失败）: {last_err[:300]}")
