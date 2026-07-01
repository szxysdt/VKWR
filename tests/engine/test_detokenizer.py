from __future__ import annotations

from vkwr.engine.detokenizer import IncrementalDetokenizer
from vkwr.engine.request import SamplingParams


class _MockTokenizer:
    """Minimal tokenizer mock that maps token IDs to specific bytes."""

    def __init__(self, id2bytes: dict[int, bytes], special_ids: set[int] | None = None):
        self._id2bytes = id2bytes
        self._special_ids = special_ids or {0}

    def get_token_bytes(self, token_id: int) -> bytes:
        return self._id2bytes.get(token_id, b"")

    def is_special(self, token_id: int) -> bool:
        return token_id in self._special_ids

    @property
    def all_special_ids(self) -> set[int]:
        return self._special_ids


class TestIncrementalDetokenizerBasic:
    def test_single_byte_tokens(self):
        tok = _MockTokenizer({1: b"H", 2: b"i", 3: b"!"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 2, 3], False)
        assert det.output_text == "Hi!"
        assert det.get_new_text() == "Hi!"

    def test_empty_update(self):
        tok = _MockTokenizer({1: b"a"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)
        result = det.update([], False)
        assert result is None
        assert det.get_new_text() == ""

    def test_multi_step_decode(self):
        tok = _MockTokenizer({1: b"Hello", 2: b" ", 3: b"World"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        assert det.output_text == "Hello"

        det.update([2], False)
        assert det.output_text == "Hello "

        det.update([3], False)
        assert det.output_text == "Hello World"

    def test_tokenizer_none(self):
        sp = SamplingParams()
        det = IncrementalDetokenizer(None, sp)
        det.update([1, 2, 3], False)
        assert det.output_text == ""
        assert det.get_new_text() == ""
        assert det._output_token_count == 3


class TestIncrementalDetokenizerUTF8:
    def test_two_byte_utf8_across_two_tokens(self):
        # "e" = 0x65, combining accent = 0xCC 0x81 → "é"
        # Or simpler: 0xC3 + 0xA9 → "é"
        tok = _MockTokenizer({100: b"\xc3", 101: b"\xa9"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        det.update([100], False)
        assert det.output_text == ""
        assert len(det._byte_buffer) == 1

        det.update([101], False)
        assert det.output_text == "é"
        assert len(det._byte_buffer) == 0

    def test_three_byte_utf8_across_three_tokens(self):
        # "α" in UTF-8 is 0xCE 0xB1 (2 bytes). For 3-byte: "α" = 0xCE 0xB1
        # "你" = 0xE4 0xBD 0xA0
        tok = _MockTokenizer({1: b"\xe4", 2: b"\xbd", 3: b"\xa0"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        assert det.output_text == ""

        det.update([2], False)
        assert det.output_text == ""

        det.update([3], False)
        assert det.output_text == "你"

    def test_four_byte_utf8_across_four_tokens(self):
        # U+1F600 = 0xF0 0x9F 0x98 0x80
        tok = _MockTokenizer({1: b"\xf0", 2: b"\x9f", 3: b"\x98", 4: b"\x80"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 2], False)
        assert det.output_text == ""

        det.update([3, 4], False)


class TestIncrementalDetokenizerSkipSpecial:
    def test_skip_eos_token(self):
        tok = _MockTokenizer({1: b"Hi", 0: b""}, special_ids={0})
        sp = SamplingParams(skip_special_tokens=True)
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 0], False)
        assert det.output_text == "Hi"

    def test_include_special_token_when_disabled(self):
        tok = _MockTokenizer({1: b"Hi", 0: b"<EOS>"}, special_ids={0})
        sp = SamplingParams(skip_special_tokens=False)
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 0], False)
        assert det.output_text == "Hi<EOS>"


class TestIncrementalDetokenizerStopStrings:
    def test_text_level_stop_match(self):
        tok = _MockTokenizer({1: b"Hello", 2: b" ---", 3: b" world"})
        sp = SamplingParams(stop=["---"])
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        result = det.update([2], False)
        assert result == "---"
        assert det.output_text == "Hello "

    def test_text_level_stop_no_match(self):
        tok = _MockTokenizer({1: b"Hello", 2: b" world"})
        sp = SamplingParams(stop=["---"])
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        result = det.update([2], False)
        assert result is None
        assert det.output_text == "Hello world"

    def test_stop_include_in_output(self):
        tok = _MockTokenizer({1: b"Hello", 2: b" ---", 3: b" world"})
        sp = SamplingParams(stop=["---"], include_stop_str_in_output=True)
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        result = det.update([2, 3], False)
        assert result == "---"
        assert det.output_text == "Hello ---"

    def test_byte_level_stop_match(self):
        # Stop string "---" = 0x2d 0x2d 0x2d. If the buffer contains an
        # incomplete multi-byte byte (0xc3) followed by the stop string bytes,
        # _try_decode_buffer cannot decode past 0xc3, so new_text_len == 0
        # and the text-level check is gated. Only byte-level can find it.
        tok = _MockTokenizer(
            {
                1: b"abc",
                2: b"\xc3",
                3: b"\x2d\x2d\x2d",
            }
        )
        sp = SamplingParams(stop=["---"])
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 2], False)
        assert det.output_text == "abc"
        assert det._byte_buffer == bytearray(b"\xc3")

        result = det.update([3], False)
        assert result == "---"
        assert det.output_text == "abc"

    def test_stop_strings_empty_list(self):
        tok = _MockTokenizer({1: b"Hi ---"})
        sp = SamplingParams(stop=[])
        det = IncrementalDetokenizer(tok, sp)
        result = det.update([1], False)
        assert result is None

    def test_stop_strings_none(self):
        tok = _MockTokenizer({1: b"Hi ---"})
        sp = SamplingParams(stop=None)
        det = IncrementalDetokenizer(tok, sp)
        result = det.update([1], False)
        assert result is None

    def test_stop_string_across_byte_buffer_boundary(self):
        # "pre" + "\xc3" → buffer has \xc3
        # "\xa9\x2d\x2d" → decodes "é", buffer has "\x2d\x2d" (partial "---")
        # "\x2d" → buffer has "---" → byte-level match
        tok = _MockTokenizer(
            {
                1: b"pre",
                2: b"\xc3",
                3: b"\xa9\x2d\x2d",
                4: b"\x2d",
            }
        )
        sp = SamplingParams(stop=["---"])
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 2], False)
        det.update([3], False)
        result = det.update([4], False)
        assert result == "---"


class TestIncrementalDetokenizerMinTokens:
    def test_min_tokens_prevents_stop(self):
        tok = _MockTokenizer({1: b"Hi", 2: b" ---"})
        sp = SamplingParams(stop=["---"], min_tokens=5)
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        result = det.update([2], False)
        assert result is None

    def test_min_tokens_exhausted_then_stop(self):
        tok = _MockTokenizer(
            {
                1: b"a",
                2: b"b",
                3: b"c",
                4: b" ---",
                5: b"x",
            }
        )
        sp = SamplingParams(stop=["---"], min_tokens=3)
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 2, 3], False)
        assert det._output_token_count == 3

        result = det.update([4], False)
        assert result == "---"

    def test_min_tokens_exhausted_in_same_update(self):
        # min_tokens=2, update with [1, 2, 3] where token 3 produces stop
        tok = _MockTokenizer({1: b"a", 2: b"b", 3: b" ---"})
        sp = SamplingParams(stop=["---"], min_tokens=2)
        det = IncrementalDetokenizer(tok, sp)

        result = det.update([1, 2, 3], False)
        assert result == "---"


class TestIncrementalDetokenizerDelta:
    def test_delta_mode(self):
        tok = _MockTokenizer({1: b"Hello", 2: b" ", 3: b"World"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        text1 = det.get_next_output_text(finished=False, delta=True)
        assert text1 == "Hello"

        det.update([2], False)
        text2 = det.get_next_output_text(finished=False, delta=True)
        assert text2 == " "

        det.update([3], False)
        text3 = det.get_next_output_text(finished=False, delta=True)
        assert text3 == "World"

    def test_cumulative_mode(self):
        tok = _MockTokenizer({1: b"Hello", 2: b" ", 3: b"World"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        det.update([1], False)
        assert det.get_next_output_text(finished=False, delta=False) == "Hello"

        det.update([2], False)
        assert det.get_next_output_text(finished=False, delta=False) == "Hello "

        det.update([3], False)
        assert det.get_next_output_text(finished=False, delta=False) == "Hello World"

    def test_delta_no_overlap(self):
        tok = _MockTokenizer({1: b"a", 2: b"b", 3: b"c", 4: b"d", 5: b"e"})
        sp = SamplingParams()
        det = IncrementalDetokenizer(tok, sp)

        parts = []
        for i in [1, 2, 3, 4, 5]:
            det.update([i], False)
            parts.append(det.get_next_output_text(finished=False, delta=True))

        assert "".join(parts) == "abcde"

    def test_stop_buffer_excluded_not_finished(self):
        tok = _MockTokenizer({1: b"Hello --- world"})
        sp = SamplingParams(stop=["---"])
        det = IncrementalDetokenizer(tok, sp)
        det.update([1], False)

        cum = det.get_next_output_text(finished=False, delta=False)
        assert "--- w" not in cum


class TestIncrementalDetokenizerStopTerminated:
    def test_stop_terminated_skips_last_token_decode(self):
        tok = _MockTokenizer({1: b"Hi", 2: b"<STOP>"})
        sp = SamplingParams(skip_special_tokens=False, include_stop_str_in_output=False)
        det = IncrementalDetokenizer(tok, sp)

        result = det.update([1, 2], stop_terminated=True)
        assert result is None
        assert det.output_text == "Hi"
        assert det._output_token_count == 2

    def test_stop_terminated_include_stop_str(self):
        tok = _MockTokenizer({1: b"Hi", 2: b"<STOP>"})
        sp = SamplingParams(
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )
        det = IncrementalDetokenizer(tok, sp)

        det.update([1, 2], stop_terminated=True)
        assert det.output_text == "Hi<STOP>"
