# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install a bounded argument converter for vLLM 0.24's native Qwen parser engine."""

import importlib.util
from functools import wraps
from pathlib import Path

from vllm.parser import qwen3
from vllm.parser.engine.parser_engine import ParserEngine


spec = importlib.util.spec_from_file_location("i2c_qwen3_arguments", Path(__file__).with_name("qwen3_arguments.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
qwen3._qwen3_arg_converter = module.convert_qwen3_arguments
qwen3.qwen3_config.cache_clear()

original_single_pass = ParserEngine._single_pass_parse


@wraps(original_single_pass)
def bounded_single_pass(self, *args, **kwargs):
    """Full responses need final arguments, not repeated streaming JSON deltas."""
    if self.parser_engine_config.name != "qwen3":
        return original_single_pass(self, *args, **kwargs)
    previous = self._stream_arg_deltas
    self._stream_arg_deltas = False
    try:
        return original_single_pass(self, *args, **kwargs)
    finally:
        self._stream_arg_deltas = previous


ParserEngine._single_pass_parse = bounded_single_pass
