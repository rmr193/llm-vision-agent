"""
LLM reasoning agent.

Talks to a local vLLM server (started with `--served-model-name` matching
`LLM.model`) via its OpenAI-compatible `/v1/chat/completions` endpoint. vLLM
is run with the ROCm backend so inference happens on the AMD GPU; from this
client's point of view it is just an OpenAI-shaped HTTP API, which keeps this
module simple and backend-agnostic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger
from openai import OpenAI

from src.config import LLM
from src.llm.prompts import SYSTEM_PROMPT, build_user_message

VALID_ACTIONS = {"click", "move", "type_text", "key", "scroll", "wait", "done"}


@dataclass
class AgentAction:
    reasoning: str
    action: str
    target_element_id: Optional[int] = None
    coordinates: Optional[List[int]] = None
    text: Optional[str] = None
    key_name: Optional[str] = None
    scroll_amount: Optional[int] = None
    done_summary: Optional[str] = None

    @classmethod
    def from_json(cls, payload: dict) -> "AgentAction":
        action = payload.get("action")
        if action not in VALID_ACTIONS:
            raise ValueError(f"LLM returned invalid action: {action!r}")
        return cls(
            reasoning=payload.get("reasoning", ""),
            action=action,
            target_element_id=payload.get("target_element_id"),
            coordinates=payload.get("coordinates"),
            text=payload.get("text"),
            key_name=payload.get("key_name"),
            scroll_amount=payload.get("scroll_amount"),
            done_summary=payload.get("done_summary"),
        )

    def to_history_line(self) -> str:
        parts = [f"action={self.action}"]
        if self.target_element_id is not None:
            parts.append(f"target_element_id={self.target_element_id}")
        if self.coordinates:
            parts.append(f"coordinates={self.coordinates}")
        if self.text:
            parts.append(f"text={self.text!r}")
        if self.key_name:
            parts.append(f"key={self.key_name}")
        if self.scroll_amount is not None:
            parts.append(f"scroll_amount={self.scroll_amount}")
        return " | ".join(parts)


class LLMAgent:
    """Thin client around a local vLLM (ROCm) server for step-by-step planning."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.model = model or LLM.model
        self.client = OpenAI(base_url=base_url or LLM.base_url, api_key="not-needed-for-local-vllm")

    def next_action(self, goal: str, elements: List[dict], history: List[str]) -> AgentAction:
        elements_json = json.dumps(elements, indent=2)
        history_text = "\n".join(f"{i+1}. {h}" for i, h in enumerate(history))
        user_message = build_user_message(goal, elements_json, history_text)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=LLM.temperature,
            max_tokens=LLM.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw_content = response.choices[0].message.content
        logger.debug(f"Raw LLM response: {raw_content}")

        try:
            payload = json.loads(raw_content)
            return AgentAction.from_json(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"Failed to parse LLM action, defaulting to 'wait'. Error: {exc}")
            return AgentAction(
                reasoning=f"Fallback due to parse error: {exc}",
                action="wait",
            )
