"""Kimi K3 tokenizer.

The release ships a tiktoken BPE (`tiktoken.model`, ~2.8 MB of base64 rank
lines), a `tokenization_kimi.py` wrapper and a `tokenizer_config.json`. This is a
thin loader over the ranks plus the special-token ids, which is all training
needs: Megatron wants `vocab_size`, `bos`, `eos`, `pad` and a way to turn text
into ids.

The ids are **not** derived from the merge file -- they come from `config.json`
(`bos 163584`, `eos 163586`, `pad 163839`, `media_placeholder 163605`) and are
asserted rather than assumed, because a shifted special-token id is the kind of
error that trains happily and produces garbage at generation time.
"""

from pathlib import Path
from typing import Dict, List, Optional

VOCAB_SIZE = 163840
BOS_TOKEN_ID = 163584
EOS_TOKEN_ID = 163586
PAD_TOKEN_ID = 163839
MEDIA_PLACEHOLDER_TOKEN_ID = 163605

#: XTML control tokens named in the plan's ground truth.
XTML_SPECIALS = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")


class K3Tokenizer:
    """Minimal tiktoken-rank tokenizer with K3's special ids."""

    def __init__(self, model_path: Optional[str] = None):
        self.vocab_size = VOCAB_SIZE
        self.bos_token_id = BOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self.media_placeholder_token_id = MEDIA_PLACEHOLDER_TOKEN_ID
        self._ranks: Dict[bytes, int] = {}
        if model_path:
            self.load_ranks(model_path)

    def load_ranks(self, model_path: str) -> int:
        """Read the base64 `<token> <rank>` lines the release ships."""
        import base64

        ranks: Dict[bytes, int] = {}
        for line in Path(model_path).read_text().splitlines():
            if not line.strip():
                continue
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
        self._ranks = ranks
        return len(ranks)

    @property
    def num_base_tokens(self) -> int:
        return len(self._ranks)

    def special_token_ids(self) -> Dict[str, int]:
        return {
            "bos": self.bos_token_id,
            "eos": self.eos_token_id,
            "pad": self.pad_token_id,
            "media_placeholder": self.media_placeholder_token_id,
        }

    def check_ids_against_config(self, config: dict) -> List[str]:
        """Return the mismatches between our ids and a released `config.json`."""
        expected = {
            "bos": config.get("bos_token_id"),
            "eos": config.get("eos_token_id"),
            "pad": config.get("pad_token_id"),
            "media_placeholder": config.get("media_placeholder_token_id"),
        }
        return [
            f"{name}: ours {ours} vs config {expected[name]}"
            for name, ours in self.special_token_ids().items()
            if expected[name] is not None and expected[name] != ours
        ]
