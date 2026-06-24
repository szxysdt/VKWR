from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.engine.request import SamplingParams
    from vkwr.tokenizers.rwkv7 import RWKVTokenizer


class IncrementalDetokenizer:
    """Incremental detokenizer for RWKV tokenizer.

    Per-request state that incrementally decodes new token IDs into text,
    handles skip_special_tokens, and stop strings detection.

    Aligned with vLLM's BaseIncrementalDetokenizer but simplified for
    RWKV's byte-level tokenizer.
    """

    def __init__(
        self,
        tokenizer: RWKVTokenizer | None,
        sampling_params: SamplingParams,
    ):
        self.tokenizer = tokenizer
        self.output_text: str = ""
        self._byte_buffer: bytearray = bytearray()

        # Track what was produced by last update() call
        self._last_new_text: str = ""

        # Output token count (excludes prompt).
        self._output_token_count: int = 0

        # Sampling params
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.min_tokens = sampling_params.min_tokens

        # Stop strings
        self.stop = sampling_params.stop or []
        self.include_stop_str_in_output = sampling_params.include_stop_str_in_output

        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0

        # Pre-compute max stop string byte length for byte-level scan.
        self._max_stop_bytes = max(len(s.encode("utf-8")) for s in self.stop) if self.stop else 0

        # Text output offset for delta mode
        self._last_output_text_offset = 0

        # min_tokens_stop_check_offset: character offset into output_text that
        # marks the boundary of text produced by the first min_tokens output
        # tokens.
        self._min_tokens_stop_check_offset: int = 0

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """Decode new tokens incrementally and check stop strings.

        Returns matched stop string or None.
        """
        if not new_token_ids:
            self._last_new_text = ""
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            new_token_ids_for_decode = new_token_ids[:-1]
        else:
            new_token_ids_for_decode = new_token_ids

        stop_check_offset = len(self.output_text)
        for tid in new_token_ids_for_decode:
            self._output_token_count += 1
            if self.tokenizer is None:
                continue
            if self.skip_special_tokens and self.tokenizer.is_special(tid):
                continue
            token_bytes = self.tokenizer.get_token_bytes(tid)
            self._byte_buffer.extend(token_bytes)
            self._try_decode_buffer()
            if self.min_tokens and self._output_token_count <= self.min_tokens:
                stop_check_offset = len(self.output_text)
                self._min_tokens_stop_check_offset = len(self.output_text)

        # Account for the skipped stop token in the count.
        if stop_terminated and not self.include_stop_str_in_output:
            self._output_token_count += 1

        new_text_len = len(self.output_text) - stop_check_offset
        self._last_new_text = self.output_text[len(self.output_text) - max(0, new_text_len) :]

        # Check stop strings only after min_tokens output tokens generated
        matched_stop = None
        if self._output_token_count > self.min_tokens and self.tokenizer is not None:
            matched_stop = self._check_stop_strings(new_text_len)
        return matched_stop

    def _try_decode_buffer(self) -> None:
        """Decode complete UTF-8 from _byte_buffer, leave incomplete bytes."""
        try:
            decoded = bytes(self._byte_buffer).decode("utf-8")
            self.output_text += decoded
            self._byte_buffer.clear()
        except UnicodeDecodeError as e:
            complete = bytes(self._byte_buffer[: e.start]).decode("utf-8")
            self.output_text += complete
            self._byte_buffer = bytearray(self._byte_buffer[e.start :])

    def get_new_text(self) -> str:
        """Return text produced by the last update() call.

        May be empty even when new token_ids were produced,
        if the tokens' bytes are held in the incomplete-UTF-8 buffer.
        """
        return self._last_new_text

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """Return cumulative or delta text, respecting stop buffer.

        If delta=True, return only text since last call.
        If finished=True, return full text (including stop buffer portion).
        """
        buffer_length = 0 if finished else self.stop_buffer_length
        full_text = self.output_text

        if not delta:
            return full_text[:-buffer_length] if buffer_length else full_text

        length = len(full_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return full_text[last_offset:length]
        return ""

    def _check_stop_strings(self, new_text_len: int) -> str | None:
        """Check stop strings on both text and byte level."""
        if not self.stop:
            return None

        text = self.output_text

        # Text-level check
        if new_text_len > 0:
            for stop_str in self.stop:
                stop_len = len(stop_str)
                search_start = len(text) - new_text_len - stop_len + 1
                if search_start < 0:
                    search_start = 0
                idx = text.find(stop_str, search_start)
                if idx != -1:
                    if not self.include_stop_str_in_output:
                        self.output_text = text[:idx]
                    else:
                        end = idx + stop_len
                        if end < len(text):
                            self.output_text = text[:end]
                    return stop_str

        # Byte-level check
        if self._byte_buffer:
            full_bytes = text.encode("utf-8") + bytes(self._byte_buffer)
            min_tokens_bytes = len(text[: self._min_tokens_stop_check_offset].encode("utf-8"))
            for stop_str in self.stop:
                stop_bytes = stop_str.encode("utf-8")
                search_start = max(
                    min_tokens_bytes,
                    len(full_bytes) - len(stop_bytes) - self._max_stop_bytes,
                )
                idx = full_bytes.find(stop_bytes, search_start)
                if idx != -1:
                    if not self.include_stop_str_in_output:
                        # Decode with errors="ignore" because full_bytes[:idx] may
                        # contain incomplete UTF-8 bytes (e.g., 0xc3 left in buffer).
                        # These bytes are safely dropped since the request is finished.
                        self.output_text = full_bytes[:idx].decode("utf-8", errors="ignore")
                    else:
                        self.output_text = full_bytes[: idx + len(stop_bytes)].decode("utf-8", errors="ignore")
                    self._byte_buffer.clear()
                    return stop_str

        return None
