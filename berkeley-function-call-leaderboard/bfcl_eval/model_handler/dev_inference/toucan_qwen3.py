"""Toucan 1k SFT Qwen3 handler for BFCL/gorilla.

Installed under:
    bfcl_eval/model_handler/dev_inference/toucan_qwen3.py

Toucan configs use multiple raw tool-call encodings; the rebuttal SFT export
normalizes them to Qwen3 XML, so evaluation should use QwenFCHandler.
"""

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


class ToucanQwen3Handler(QwenFCHandler):
    """Qwen3 XML handler for Toucan-normalized SFT checkpoints."""

    dataset_name = "toucan"
