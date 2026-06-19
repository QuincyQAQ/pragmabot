"""Qwen VL API client — OpenAI-compatible interface for PragmaBot."""

import time, json, re, urllib.request
from typing import List, Type


class QwenClient:
    """Mimics OpenAI SDK interface so PragmaBot's VLMClient works with Qwen VL."""

    def __init__(self, api_key: str, chat_model: str = "qwen-vl-plus",
                 embed_model: str = "text-embedding-v4",
                 chat_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 embed_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"):
        self._api_key = api_key
        self._chat_url = chat_url
        self._embed_url = embed_url
        self._chat_model = chat_model
        self._embed_model = embed_model
        self.last_raw_text = ""

    # ---- OpenAI client shape ----
    @property
    def chat(self):
        return _ChatProxy(self)

    @property
    def embeddings(self):
        return _EmbedProxy(self)


class _ChatProxy:
    def __init__(self, parent: QwenClient):
        self._p = parent

    @property
    def completions(self):
        return _CompletionsProxy(self._p)


class _CompletionsProxy:
    def __init__(self, parent: QwenClient):
        self._p = parent

    def parse(self, *, model, messages, response_format, temperature):
        # Append JSON instruction
        msgs = list(messages)
        msgs.append({"role": "system",
                      "content": "Reply with valid JSON only. No markdown fences, no extra text."})

        data = json.dumps({
            "model": self._p._chat_model,
            "messages": msgs,
            "max_tokens": 800,
            "temperature": temperature,
        }).encode()

        req = urllib.request.Request(self._p._chat_url, data=data, headers={
            "Authorization": f"Bearer {self._p._api_key}",
            "Content-Type": "application/json",
        })
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = json.loads(r.read())

        text = raw["choices"][0]["message"]["content"]
        self._p.last_raw_text = text
        prompt_tokens = raw.get("usage", {}).get("prompt_tokens", 0)

        # Parse JSON from Qwen's text response
        parsed_dict = _extract_json(text, response_format)
        _fill_defaults(parsed_dict, response_format)
        parsed_obj = response_format(**parsed_dict)

        return _FakeResponse(parsed_obj, text, prompt_tokens)


class _EmbedProxy:
    def __init__(self, parent: QwenClient):
        self._p = parent

    def create(self, *, model, input):
        texts = input if isinstance(input, list) else [input]
        data = json.dumps({"model": self._p._embed_model, "input": texts}).encode()
        req = urllib.request.Request(self._p._embed_url, data=data, headers={
            "Authorization": f"Bearer {self._p._api_key}",
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = json.loads(r.read())
        embs = [d["embedding"] for d in raw["data"]]
        return type("EmbResp", (), {
            "data": [type("EmbData", (), {"embedding": e})() for e in embs],
            "usage": type("Usage", (), {"prompt_tokens": raw.get("usage", {}).get("total_tokens", 0)})(),
        })()


class _FakeResponse:
    def __init__(self, parsed, content, prompt_tokens):
        self.choices = [type("Choice", (), {
            "message": type("Msg", (), {"parsed": parsed, "content": content})()
        })()]
        self.usage = type("Usage", (), {"prompt_tokens": prompt_tokens})()


# ---- JSON parsing helpers ----

# Map Qwen's casual field names to PragmaBot's NextBestAction field names
_FIELD_ALIASES = {
    "action": "chosen_action",
    "object": "target_object",
    "target": "target_object",
    "skill": "chosen_skill",
    "reasoning": "chain_of_thought_reasoning",
    "thinking": "chain_of_thought_reasoning",
    "thought": "chain_of_thought_reasoning",
    "description": "scene_description",
    "scene": "scene_description",
    "success": "is_action_successful",
    "action_successful": "is_action_successful",
    "action_success": "is_action_successful",
    "complete": "is_task_completed",
    "task_complete": "is_task_completed",
    "task_success": "is_task_completed",
    "task_done": "is_task_completed",
    "done": "is_task_completed",
    "change": "scene_description",
    "scene_change_desc": "scene_description",
    "why": "scene_description",
}


def _remap(d: dict) -> dict:
    """Rename keys using _FIELD_ALIASES."""
    for old, new in _FIELD_ALIASES.items():
        if old in d and new not in d:
            d[new] = d.pop(old)
    return d


def _extract_json(text: str, response_format: Type) -> dict:
    """Try to extract a JSON dict from VLM text output."""
    # 1. ```json ... ```
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try: return _remap(json.loads(m.group(1)))
        except: pass
    # 2. Bare JSON
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        s = m.group()
        depth, start = 0, s.find('{')
        for i, c in enumerate(s):
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try: return _remap(json.loads(s[start:i + 1]))
                    except: break
        try: return _remap(json.loads(s))
        except: pass
    # 3. Fallback: plain text into first string field
    first_str = next((k for k, v in response_format.model_fields.items() if v.annotation == str), None)
    if first_str:
        return {first_str: text.strip()}
    raise ValueError(f"Cannot parse: {text[:200]}")


def _infer_skill(d: dict) -> str:
    """Guess skill from action text if not explicitly given."""
    text = (d.get("chosen_action", "")).lower()
    for skill in ["pick", "place", "push"]:
        if skill in text:
            return skill
    return "push"  # default


def _fill_defaults(d: dict, response_format: Type) -> None:
    # Infer skill from action text if missing
    if "chosen_skill" not in d or not d.get("chosen_skill"):
        d["chosen_skill"] = _infer_skill(d)
    """Fill missing fields with sensible defaults so Pydantic doesn't crash."""
    for k, v in response_format.model_fields.items():
        if k not in d or d[k] is None or d[k] == "":
            anno = v.annotation
            if hasattr(anno, '__members__'):  # Enum
                d[k] = list(anno.__members__.values())[0].value
            elif hasattr(anno, '__args__') and type(None) in getattr(anno, '__args__', ()):
                d[k] = None
            elif anno == str:
                d[k] = "unknown"
            elif anno == bool:
                d[k] = False
            elif anno in (int, float):
                d[k] = 0
            else:
                d[k] = None
