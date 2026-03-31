from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_SPLIT_RE = re.compile(GPT2_SPLIT_PATTERN)

Token = bytes
Pair = tuple[Token, Token]
TokenSeq = tuple[Token, ...]


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs), strict=True))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def _special_token_regex(special_tokens: tuple[str, ...]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    ordered = sorted(special_tokens, key=len, reverse=True)
    return re.compile("|".join(re.escape(token) for token in ordered))


def split_on_special_tokens(text: str, special_tokens: Iterable[str]) -> Iterator[tuple[bool, str]]:
    special_tokens_tuple = tuple(special_tokens)
    pattern = _special_token_regex(special_tokens_tuple)
    if pattern is None:
        if text:
            yield False, text
        return

    last_end = 0
    for match in pattern.finditer(text):
        if match.start() > last_end:
            yield False, text[last_end : match.start()]
        yield True, match.group(0)
        last_end = match.end()
    if last_end < len(text):
        yield False, text[last_end:]


def pretokenize_text(text: str) -> Iterator[bytes]:
    for match in GPT2_SPLIT_RE.finditer(text):
        yield match.group(0).encode("utf-8")


def merge_pair_in_word(word: TokenSeq, pair: Pair) -> TokenSeq:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def _word_from_bytes(token_bytes: bytes) -> TokenSeq:
    return tuple(bytes([b]) for b in token_bytes)


def _iter_pairs(word: TokenSeq) -> Iterator[Pair]:
    for left, right in zip(word, word[1:], strict=False):
        yield left, right


def _add_word_stats(
    word: TokenSeq,
    freq: int,
    pair_counts: Counter[Pair],
    pair_to_words: defaultdict[Pair, set[TokenSeq]],
) -> None:
    for pair in _iter_pairs(word):
        pair_counts[pair] += freq
        pair_to_words[pair].add(word)


def _remove_word_stats(
    word: TokenSeq,
    freq: int,
    pair_counts: Counter[Pair],
    pair_to_words: defaultdict[Pair, set[TokenSeq]],
) -> None:
    for pair in _iter_pairs(word):
        pair_counts[pair] -= freq
        if pair_counts[pair] <= 0:
            pair_counts.pop(pair, None)
        words = pair_to_words.get(pair)
        if words is None:
            continue
        words.discard(word)
        if not words:
            pair_to_words.pop(pair, None)


def _collect_word_freqs(text: str, special_tokens: list[str]) -> Counter[TokenSeq]:
    word_freqs: Counter[TokenSeq] = Counter()
    for is_special, segment in split_on_special_tokens(text, special_tokens):
        if is_special:
            continue
        for pretoken in pretokenize_text(segment):
            word = _word_from_bytes(pretoken)
            if word:
                word_freqs[word] += 1
    return word_freqs


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    token_to_id = {token: idx for idx, token in vocab.items()}

    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in token_to_id:
            token_id = len(vocab)
            vocab[token_id] = token_bytes
            token_to_id[token_bytes] = token_id

    if len(vocab) >= vocab_size:
        limited_vocab = {idx: vocab[idx] for idx in range(vocab_size)}
        return limited_vocab, []

    text = Path(input_path).read_text(encoding="utf-8")
    word_freqs = _collect_word_freqs(text, special_tokens)

    pair_counts: Counter[Pair] = Counter()
    pair_to_words: defaultdict[Pair, set[TokenSeq]] = defaultdict(set)
    for word, freq in word_freqs.items():
        _add_word_stats(word, freq, pair_counts, pair_to_words)

    merges: list[Pair] = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        affected_words = list(pair_to_words.get(best_pair, ()))
        if not affected_words:
            break

        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        for old_word in affected_words:
            freq = word_freqs.pop(old_word, 0)
            if freq == 0:
                continue
            _remove_word_stats(old_word, freq, pair_counts, pair_to_words)
            new_word = merge_pair_in_word(old_word, best_pair)
            word_freqs[new_word] += freq
            _add_word_stats(new_word, freq, pair_counts, pair_to_words)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.id_to_token = dict(vocab)
        self.token_to_id = {token: idx for idx, token in self.id_to_token.items()}
        self.merges = list(merges)
        self.pair_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = list(special_tokens or [])

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                token_id = len(self.id_to_token)
                self.id_to_token[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id

        self.special_token_to_id = {token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens}
        self._special_tokens_tuple = tuple(self.special_tokens)
        self._max_special_token_len = max((len(token) for token in self.special_tokens), default=0)
        self._special_token_prefixes = {
            token[:prefix_len]
            for token in self.special_tokens
            for prefix_len in range(1, len(token))
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        byte_decoder = unicode_to_bytes()
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)
        vocab = {
            token_id: bytes(byte_decoder[ch] for ch in token_string)
            for token_string, token_id in serialized_vocab.items()
        }

        merges: list[Pair] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                cleaned = line.rstrip("\n")
                if not cleaned:
                    continue
                pieces = cleaned.split(" ")
                if len(pieces) != 2:
                    continue
                left, right = pieces
                merges.append(
                    (
                        bytes(byte_decoder[ch] for ch in left),
                        bytes(byte_decoder[ch] for ch in right),
                    )
                )
        return cls(vocab, merges, special_tokens)

    def _merge_pretoken(self, pretoken: bytes) -> list[bytes]:
        pieces = [bytes([b]) for b in pretoken]
        while len(pieces) >= 2:
            best_rank: int | None = None
            best_pair: Pair | None = None
            for left, right in zip(pieces, pieces[1:], strict=False):
                pair = (left, right)
                rank = self.pair_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            merged: list[bytes] = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and pieces[i] == best_pair[0] and pieces[i + 1] == best_pair[1]:
                    merged.append(pieces[i] + pieces[i + 1])
                    i += 2
                else:
                    merged.append(pieces[i])
                    i += 1
            pieces = merged
        return pieces

    def _encode_pretoken(self, pretoken: bytes) -> Iterator[int]:
        for token in self._merge_pretoken(pretoken):
            yield self.token_to_id[token]

    def _encode_ordinary_text(self, text: str) -> Iterator[int]:
        for pretoken in pretokenize_text(text):
            yield from self._encode_pretoken(pretoken)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for is_special, segment in split_on_special_tokens(text, self._special_tokens_tuple):
            if is_special:
                ids.append(self.special_token_to_id[segment])
            else:
                ids.extend(self._encode_ordinary_text(segment))
        return ids

    def _longest_partial_special_suffix(self, text: str) -> int:
        if not self._special_token_prefixes:
            return 0
        max_len = min(len(text), self._max_special_token_len - 1)
        for suffix_len in range(max_len, 0, -1):
            if text[-suffix_len:] in self._special_token_prefixes:
                return suffix_len
        return 0

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            partial_special_len = self._longest_partial_special_suffix(buffer)
            if partial_special_len:
                processable = buffer[:-partial_special_len]
                next_buffer = buffer[-partial_special_len:]
            else:
                processable = buffer
                next_buffer = ""

            if not processable:
                buffer = next_buffer
                continue

            segments = list(split_on_special_tokens(processable, self._special_tokens_tuple))
            if partial_special_len:
                for is_special, segment in segments:
                    if is_special:
                        yield self.special_token_to_id[segment]
                    else:
                        yield from self._encode_ordinary_text(segment)
                buffer = next_buffer
                continue

            if segments and not segments[-1][0]:
                final_text = segments[-1][1]
                stable_segments = segments[:-1]
            else:
                final_text = ""
                stable_segments = segments

            for is_special, segment in stable_segments:
                if is_special:
                    yield self.special_token_to_id[segment]
                else:
                    yield from self._encode_ordinary_text(segment)

            last_match: bytes | None = None
            for match in GPT2_SPLIT_RE.finditer(final_text):
                current = match.group(0).encode("utf-8")
                if last_match is not None:
                    yield from self._encode_pretoken(last_match)
                last_match = current
            buffer = next_buffer + ("" if last_match is None else last_match.decode("utf-8"))

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.id_to_token[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
