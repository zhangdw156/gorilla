"""ToolACE 1k SFT Qwen3 handler for BFCL/gorilla.

Installed under:
    bfcl_eval/model_handler/dev_inference/toolace_qwen3.py

ToolACE source calls are function-call strings such as
``[tool(arg=value)]`` during data normalization, but trained checkpoints are
queried with the shared Qwen3 XML ``<tool_call>`` contract.
"""

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


class ToolACEQwen3Handler(QwenFCHandler):
    """Qwen3 XML handler for ToolACE-normalized SFT checkpoints."""

    dataset_name = "toolace"
