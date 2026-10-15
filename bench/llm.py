"""The LLM that labels the script and writes each engine's script: any OpenAI-compatible
chat-completions server (LM Studio, Ollama, llama.cpp, vLLM, or a hosted API). Standard
library only.

Settings come from the environment or `.env` at the repo root (see `.env.example`):
LLM_BASE_URL (default LM Studio's http://localhost:1234/v1), LLM_MODEL, LLM_API_KEY,
LLM_NO_THINK (append /no_think for Qwen3-style thinking models).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
CALL_TIMEOUT_S = 600


class LLMError(RuntimeError):
    pass


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=VALUE lines into os.environ (values already set win)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip().strip("\"'")
            if value:  # an empty KEY= means "not set" (e.g. HF_TOKEN= keeps the login token)
                os.environ.setdefault(key.strip(), value)


@dataclass
class LLM:
    base_url: str
    model: str
    api_key: str = ""
    no_think: bool = False
    # LM Studio's own field that really turns thinking off; a server that rejects it is asked
    # again without it, once, and never again.
    _send_enable_thinking: bool = True

    @classmethod
    def from_env(cls, base_url: str = "", model: str = "") -> LLM:
        return cls(
            base_url=base_url or os.environ.get("LLM_BASE_URL") or "http://localhost:1234/v1",
            model=model or os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
            no_think=os.environ.get("LLM_NO_THINK", "").lower() in {"1", "true", "yes", "on"},
        )

    def describe(self) -> str:
        return f"{self.model or '(server default)'} at {self.base_url}"

    def complete(self, system: str, user: str, *, max_tokens: int = 4000) -> str:
        if self.no_think:
            user = f"{user}\n\n/no_think"
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if self.model:
            body["model"] = self.model
        if self._send_enable_thinking:
            body["enable_thinking"] = False
        try:
            payload = self._post(body)
        except HTTPError as exc:
            if exc.code == 400 and self._send_enable_thinking:
                self._send_enable_thinking = False
                body.pop("enable_thinking")
                payload = self._post(body)
            else:
                raise LLMError(f"LLM server error {exc.code}: {exc.read()[:500]!r}") from exc
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError(f"LLM returned no choices: {str(payload)[:500]}")
        content = (choices[0].get("message") or {}).get("content") or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content:
            # usually hidden reasoning that used up max_tokens
            details = (payload.get("usage") or {}).get("completion_tokens_details") or {}
            raise LLMError(
                f"empty reply (finish_reason={choices[0].get('finish_reason')}, "
                f"reasoning_tokens={details.get('reasoning_tokens')}, max_tokens={max_tokens})"
            )
        return content

    def _post(self, body: dict) -> dict:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urlreq.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urlreq.urlopen(req, timeout=CALL_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError:
            raise
        except (URLError, TimeoutError) as exc:
            raise LLMError(f"cannot reach the LLM at {url}: {exc}") from exc

    def check(self) -> None:
        """Fail early with a clear message when the server isn't there."""
        self.complete("Reply with the single word: ok", "ok?", max_tokens=2000)


def unload_lm_studio(log) -> None:
    """Free the GPU for the voice models: LM Studio keeps the LLM loaded otherwise."""
    lms = shutil.which("lms") or str(Path.home() / ".lmstudio" / "bin" / "lms")
    if Path(lms).exists():
        result = subprocess.run([lms, "unload", "--all"], capture_output=True, text=True)
        log(f"lms unload --all: {(result.stdout or result.stderr).strip()[:200]}")


def parse_json(text: str, kind: type = dict):
    """The first JSON object (or array) in a reply, tolerating code fences and prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip())
    open_, close = ("{", "}") if kind is dict else ("[", "]")
    start, end = text.find(open_), text.rfind(close)
    if start == -1 or end < start:
        raise ValueError(f"no JSON {kind.__name__} in reply: {text[:200]!r}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, kind):
        raise ValueError(f"expected a JSON {kind.__name__}")
    return value
