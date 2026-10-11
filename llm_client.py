"""LLM clients used by the app.

RemoteLLM : calls the Mistral model served from Kaggle through FastAPI + ngrok.
MockLLM   : returns fake answers so you can test the UI without a GPU.
Both are LangChain LLMs, so they plug directly into prompt | llm chains.
"""
import json
import re
import zlib
from typing import Any, List, Optional

import requests
from langchain_core.language_models.llms import LLM


class RemoteLLM(LLM):
    url: str
    api_key: str = "secret123"
    max_new_tokens: int = 900
    temperature: float = 0.1
    timeout: int = 600

    @property
    def _llm_type(self) -> str:
        return "remote_mistral_ngrok"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "ngrok-skip-browser-warning": "true",
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> str:
        res = requests.post(
            self.url.rstrip("/") + "/generate",
            headers=self._headers(),
            json={
                "prompt": prompt,
                "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
            timeout=self.timeout,
        )
        res.raise_for_status()
        return res.json()["response"]

    def health(self) -> dict:
        res = requests.get(self.url.rstrip("/") + "/health", headers=self._headers(), timeout=20)
        res.raise_for_status()
        return res.json()


class MockLLM(LLM):
    """Offline stand-in for the real model (demo mode). Recognises each prompt by a marker phrase."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> str:
        title = prompt.split('titled: "', 1)[-1].split('"', 1)[0] if 'titled: "' in prompt else \
            prompt.split('the paper "', 1)[-1].split('"', 1)[0]
        seed = zlib.crc32(title.encode()) % 15

        if "routing agent" in prompt:
            req = prompt.split("User request:", 1)[-1].split("Return ONLY", 1)[0].lower()
            tool = ("compare" if "compar" in req or "differ" in req else
                    "chart" if "chart" in req or "plot" in req or "accuracy" in req else
                    "related_work" if "related work" in req or "literature" in req else
                    "graph" if "graph" in req or "connect" in req else
                    "arxiv" if "find" in req or "search" in req or "new papers" in req else "qa")
            return "```json\n" + json.dumps({"tool": tool, "query": "", "reason": "[demo] keyword routing"}) + "\n```"
        if "strict fact-checker" in prompt:
            n = len(re.findall(r"^\d+\. ", prompt.split("CLAIMS:", 1)[-1], re.M))
            verdicts = ["supported", "supported", "partial", "unsupported"]
            return "```json\n" + json.dumps(
                [{"id": i, "verdict": verdicts[(i - 1) % 4], "source": "[demo] p. 1"} for i in range(1, n + 1)]
            ) + "\n```"
        if "extract structured FACTS" in prompt:
            data = {
                "methods": [title.split()[0] + " Net", "Adam optimizer"],
                "datasets": ["ImageNet", "CIFAR-10"] if seed % 2 else ["ImageNet"],
                "tasks": ["Image classification"],
                "results": [
                    {"model": title.split()[0], "dataset": "ImageNet", "metric": "Top-1 Accuracy", "value": 75 + seed, "unit": "%"},
                    {"model": title.split()[0] + "-Large", "dataset": "ImageNet", "metric": "Top-1 Accuracy", "value": 78 + seed, "unit": "%"},
                ],
            }
            return "```json\n" + json.dumps(data, indent=2) + "\n```"
        if "Extract the following information" in prompt:
            data = {
                "paper": title,
                "methodology": "[demo] A deep learning model trained on the described data.",
                "dataset": "[demo] Public dataset described in the paper.",
                "results": f"[demo] Top-1 accuracy of {75 + seed}% on ImageNet.",
                "advantages": "- [demo] Simple pipeline\n- [demo] Strong results",
                "limitations": "- [demo] Single-center data\n- [demo] Small sample size",
            }
            return "```json\n" + json.dumps(data, indent=2) + "\n```"
        if "Related Work" in prompt:
            return ("[demo] Convolutional architectures with residual connections [1] established strong baselines, "
                    "while attention-based models [2] showed that convolutions are not strictly necessary. "
                    "Later work revisited CNN design using transformer ideas [3].\n\n"
                    "However, none of these works studies data-limited medical settings, which motivates further research.")
        if "key DIFFERENCES" in prompt:
            return "### Methodology\n- [demo] Paper A uses a CNN, paper B uses a transformer.\n\n**Why it matters:** demo text.\n\n### Research gaps\n- [demo] No external validation."
        if "Write a comparison" in prompt:
            return "## Common ground\n- [demo] Both papers address the same task.\n\n## Which to use when\n- [demo] Paper A for small data, paper B for large data."
        return ("[demo] The first paper reports 76% top-1 accuracy on ImageNet [Paper A, p. 1]. "
                "The second paper trains a transformer on a larger dataset [Paper B, p. 2]. "
                "Both papers evaluate on ImageNet with the same metric. "
                "Connect the Kaggle server to get real answers.")
