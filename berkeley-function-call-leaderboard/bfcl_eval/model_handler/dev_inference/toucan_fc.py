"""
Toucan-Qwen2.5-7B function-calling handler for local OSS inference (BFCL).

Uses the same ``<tool_call>`` / ``<tool_response>`` layout as :class:`QwenFCHandler`,
with system prompt aligned to Toucan's Hermes/NousResearch training format.

Key difference from other Qwen-style handlers:
- No base system prompt text before the ``# Tools`` section.
- Tool definitions are wrapped in ``{"type": "function", "function": ...}``.
"""

from __future__ import annotations

import json

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override


def _build_toucan_system_with_tools(function: list) -> str:
    if not function:
        return ""
    tool_lines = "\n".join(
        json.dumps({"type": "function", "function": tool}) for tool in function
    )
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user query."
        + "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        + tool_lines
        + "\n</tools>\n\nFor each function call, return a json object with function name and "
        + "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        + '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


class ToucanFCHandler(QwenFCHandler):
    """
    Qwen-style FC layout with Toucan (Hermes/NousResearch) training system prompt.

    Inherits decode_ast / decode_execute / _extract_tool_calls from QwenFCHandler
    (same ``<tool_call>`` XML output format).
    """

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            formatted_prompt += _build_toucan_system_with_tools(function)
            formatted_prompt += "<|im_end|>\n"
        else:
            if messages and messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                formatted_prompt += f"<|im_start|>{role}\n{content}"

                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or (
                            tool_call != message["tool_calls"][0]
                        ):
                            formatted_prompt += "\n"

                        if "function" in tool_call:
                            tool_call = tool_call["function"]

                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '

                        if isinstance(tool_call["arguments"], str):
                            formatted_prompt += tool_call["arguments"]
                        else:
                            formatted_prompt += json.dumps(tool_call["arguments"])

                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = (
                    messages[idx + 1]["role"] if idx < len(messages) - 1 else None
                )

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt
