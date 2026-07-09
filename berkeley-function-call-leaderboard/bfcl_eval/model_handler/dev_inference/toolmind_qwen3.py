"""ToolMind 1k SFT Qwen3 handler for BFCL/gorilla.

Installed under:
    bfcl_eval/model_handler/dev_inference/toolmind_qwen3.py

ToolMind source rows already use OpenAI-style ``tool_calls``; this handler keeps
BFCL inference on the same Qwen3 XML tool-call template used by the normalized
SFT rows.
"""

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


class ToolMindQwen3Handler(QwenFCHandler):
    """Qwen3 XML handler for ToolMind-normalized SFT checkpoints."""

    dataset_name = "toolmind"
