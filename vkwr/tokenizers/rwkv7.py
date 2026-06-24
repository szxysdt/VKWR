
from __future__ import annotations

import ast
import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VOCAB_FILE = "rwkv_vocab_v20230424.txt"
VOCAB_DOWNLOAD_URL = "https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_vocab_v20230424.txt"


class _TrieNode:
    __slots__ = ("to", "token")
    to: list[_TrieNode | None]
    token: int

    def __init__(self) -> None:
        self.to = [None] * 256
        self.token = 0

    def add(self, key: bytes, val: int) -> None:
        node = self
        for ch in key:
            nxt = node.to[ch]
            if nxt is None:
                nxt = _TrieNode()
                node.to[ch] = nxt
            node = nxt
        node.token = val + 1


class RWKVTokenizer:
    """RWKV tokenizer using a TRIE for fast greedy byte-level matching.

    Based on ChatRWKV TRIE_TOKENIZER. Loads rwkv_vocab_v20230424.txt,
    which stores each token as a Python string literal.
    """

    def __init__(self, vocab_path: str | Path | None = None) -> None:
        self._vocab_path = Path(vocab_path) if vocab_path else None
        self._idx2token: list[bytes] | None = None
        self._token2idx: dict[bytes, int] | None = None
        self._root: _TrieNode | None = None
        self._ensure_loaded()

    @property
    def eos_token_id(self) -> int:
        return 0

    def _ensure_loaded(self) -> None:
        if self._root is not None:
            return
        path = self._find_vocab()
        if path is None:
            raise FileNotFoundError(
                f"RWKV vocab file '{VOCAB_FILE}' not found.\n  Please download it and place it next to your model file:\n  {VOCAB_DOWNLOAD_URL}"
            )
        self._load_vocab(path)

    def _find_vocab(self) -> Path | None:
        if self._vocab_path is not None:
            if self._vocab_path.is_file():
                return self._vocab_path
            if self._vocab_path.is_dir():
                candidate = self._vocab_path / VOCAB_FILE
                if candidate.exists():
                    return candidate
        return None

    def _load_vocab(self, path: Path) -> None:
        idx2token: dict[int, bytes] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx = int(line[: line.index(" ")])
                token_str_literal = line[line.index(" ") : line.rindex(" ")]
                x: str | bytes = ast.literal_eval(token_str_literal)
                token_bytes = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(token_bytes, bytes)
                idx2token[idx] = token_bytes

        token2idx: dict[bytes, int] = {}
        for k, v in idx2token.items():
            token2idx[v] = k

        max_id = max(idx2token)
        idx2token_list: list[bytes] = [b""] * (max_id + 1)
        for idx, tok in idx2token.items():
            idx2token_list[idx] = tok

        root = _TrieNode()
        for tok, i in token2idx.items():
            root.add(tok, i)

        self._idx2token = idx2token_list
        self._token2idx = token2idx
        self._root = root
        logger.info("Loaded RWKV vocab from %s (%d tokens)", path, len(idx2token))

    def encode(self, text: str) -> list[int]:
        self._ensure_loaded()
        root_to = self._root.to  # type: ignore[union-attr]
        src = text.encode("utf-8")
        tokens: list[int] = []
        src_len = len(src)
        idx = 0
        while idx < src_len:
            node = root_to[src[idx]]
            token = node.token  # type: ignore[union-attr]
            end = idx + 1
            child_to = node.to  # type: ignore[union-attr]
            j = idx + 1
            while j < src_len:
                node = child_to[src[j]]
                if node is None:
                    break
                tok = node.token
                if tok:
                    token = tok
                    end = j + 1
                child_to = node.to
                j += 1
            tokens.append(token - 1)
            idx = end
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        self._ensure_loaded()
        assert self._idx2token is not None
        raw: list[bytes] = []
        for tid in token_ids:
            if tid < len(self._idx2token):
                raw.append(self._idx2token[tid])
        return b"".join(raw).decode("utf-8", errors="replace")


@functools.cache
def get_tokenizer(vocab_path: str | None = None) -> RWKVTokenizer | None:
    """Try to create a tokenizer; return None if vocab file unavailable."""
    try:
        return RWKVTokenizer(vocab_path)
    except FileNotFoundError:
        logger.warning("RWKV vocab file not found — text prompts will fail")
        return None
    except Exception:
        logger.warning("Failed to load RWKV tokenizer", exc_info=True)
        return None
