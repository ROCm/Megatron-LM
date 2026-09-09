# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers for HuggingFace ``apply_chat_template`` tokenized outputs."""

from collections.abc import Mapping

import numpy as np


def token_ids_from_chat_template_output(output):
    """Normalize ``apply_chat_template(..., tokenize=True, ...)`` to 1-D int64 token ids.

    Depending on ``transformers`` / ``tokenizers`` versions, tokenized output may be a
    ``numpy`` array, ``transformers.BatchEncoding`` (``input_ids``), a ``tokenizers.Encoding``,
    or a length-1 batch of encodings.
    """
    if isinstance(output, Mapping) and "input_ids" in output:
        return token_ids_from_chat_template_output(output["input_ids"])

    try:
        import torch

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(output, np.ndarray):
        arr = np.asarray(output, dtype=np.int64)
        if arr.ndim >= 2:
            arr = arr[0]
        return arr.reshape(-1)

    # tokenizers.Encoding (not dict-like)
    if hasattr(output, "ids") and not isinstance(output, (str, bytes, list, tuple, Mapping)):
        return np.asarray(output.ids, dtype=np.int64).reshape(-1)

    if isinstance(output, (list, tuple)):
        if len(output) == 1 and hasattr(output[0], "ids") and not isinstance(output[0], Mapping):
            return np.asarray(output[0].ids, dtype=np.int64).reshape(-1)
        arr = np.asarray(output, dtype=np.int64)
        if arr.ndim >= 2:
            arr = arr[0]
        return arr.reshape(-1)

    raise TypeError(f"Unexpected apply_chat_template output type: {type(output)}")
