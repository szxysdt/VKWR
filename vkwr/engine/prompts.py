"""RWKV chat prompt templates.

Implements the prompt format used by rwkv_lightning.  The standard
template follows ``User: ...\\n\\nAssistant: ...`` convention with
optional thinking tags and system prefix.

Reference implementations:

* ``third_party/rwkv_lightning/webui_rwkv.py``  (``build_prompt``)
* ``third_party/rwkv_lightning/API_servers/openai_routes.py``  (``format_openai_prompt``)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RWKVChatTemplate:
    """An RWKV chat prompt template.

    Attributes:
        name: Human-readable template name.
        system_prefix: Prefix for system messages.
        user_prefix: Prefix for user messages.
        assistant_prefix: Prefix for assistant messages.
        separator: Separator between turns.
        thinking_tag: Appended after assistant prefix for normal mode.
        think_tag: Appended after assistant prefix for think mode.
        stop_tokens: Default stop tokens to end generation.
    """

    name: str
    system_prefix: str = "System"
    user_prefix: str = "User"
    assistant_prefix: str = "Assistant"
    separator: str = "\n\n"
    thinking_tag: str = "<think>\n</think>\n"
    think_tag: str = "<think"
    stop_tokens: list[str] = field(default_factory=list)


# ── Built-in template (rwkv_lightning default) ─────────────────────

RWKV_CHAT_TEMPLATE = RWKVChatTemplate(
    name="rwkv-default",
    system_prefix="System",
    user_prefix="User",
    assistant_prefix="Assistant",
    stop_tokens=["\nUser:"],
)


def apply_chat_template(
    messages: list[dict[str, str]],
    *,
    template: RWKVChatTemplate | None = None,
    add_generation_prompt: bool = True,
    enable_think: bool = False,
) -> str:
    """Convert OpenAI-style messages to an RWKV prompt string.

    Args:
        messages: List of ``{"role": str, "content": str}`` dicts.
            Supported roles: ``"system"``, ``"user"``, ``"assistant"``.
            Role ``"developer"`` is treated as ``"system"``.
        template: Template to use, or ``None`` for the default.
        add_generation_prompt: If ``True``, append the assistant prefix
            so the model is primed to generate a reply.
        enable_think: If ``True``, use ``<think`` instead of reasoning tags.

    Returns:
        The assembled prompt string ready for tokenization.

    Example:
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> apply_chat_template(msgs)
        'User: Hello\\n\\nAssistant: <think>\\n</think>\\n'
    """
    tmpl = template or RWKV_CHAT_TEMPLATE
    parts: list[str] = []

    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""

        if role in ("system", "developer"):
            parts.append(f"{tmpl.system_prefix}: {content}")
        elif role == "user":
            parts.append(f"{tmpl.user_prefix}: {content}")
        elif role == "assistant":
            parts.append(f"{tmpl.assistant_prefix}: {content}")

    prompt = tmpl.separator.join(parts)

    if add_generation_prompt:
        tag = tmpl.think_tag if enable_think else tmpl.thinking_tag
        prompt = f"{prompt}{tmpl.separator}{tmpl.assistant_prefix}: {tag}"

    return prompt


def get_default_stop_tokens(
    template: RWKVChatTemplate | None = None,
) -> list[str]:
    """Return the default stop tokens for the RWKV chat template.

    Args:
        template: Template instance, or ``None`` for the default.

    Returns:
        List of stop token strings (e.g. ``["\\nUser:"]``).
    """
    tmpl = template or RWKV_CHAT_TEMPLATE
    return list(tmpl.stop_tokens)
