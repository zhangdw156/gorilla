"""APIGen-MT 1k SFT Qwen3 handler for BFCL/gorilla.

Installed under:
    bfcl_eval/model_handler/dev_inference/apigen_mt_qwen3.py

This intentionally uses BFCL's QwenFCHandler, not SalesforceQwenHandler:
SalesforceQwenHandler is the official xLAM/Salesforce handler and expects a
JSON-array tool-call format, while the ASTRA rebuttal data normalized from
APIGen-MT is trained with Qwen3's XML ``<tool_call>`` / ``<tool_response>``
chat-template contract.
"""

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


class APIGenMTQwen3Handler(QwenFCHandler):
    """Qwen3 XML handler for APIGen-MT-normalized SFT checkpoints."""

    dataset_name = "apigen_mt"
