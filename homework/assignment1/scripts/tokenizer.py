"""Assignment 1 - BPE Tokenizer 实现

基于 Stanford CS336 Assignment 1 的 BPE Tokenizer 实现。
参考：https://github.com/stanford-cs336/assignment1-basics
"""

from collections.abc import Iterable, Iterator

import regex


class Tokenizer:
    """BPE Tokenizer，兼容 GPT-2 编码结果。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # TODO: 初始化
        # - 存储 vocab、merges
        # - 构建 vocab 的反向查找表 (bytes/str → int)
        # - 构建 merge pair 到 rank 的映射（用于快速查找合并优先级）
        # - 处理 special tokens（构建 split pattern）
        self.gpt2_pat = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        )
        self.bytes2unicode = self.bytes_to_unicode()
        self.bytes2unicode_reverse = {
            v: k for k, v in self.bytes2unicode.items()
        }
        self.vocab = {
            k: "".join([self.bytes2unicode.get(b, chr(b)) for b in v])
            for k, v in vocab.items()
        }
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.merges = [
            (
                "".join([self.bytes2unicode.get(b, chr(b)) for b in k1]),
                "".join([self.bytes2unicode.get(b, chr(b)) for b in k2]),
            )
            for k1, k2 in merges
        ]
        self.special_tokens = special_tokens or []
        self.special_pattern = (
            "(" + "|".join(map(regex.escape, self.special_tokens)) + ")"
        )
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def bytes_to_unicode(self) -> dict[int, str]:
        # 可直接显示的字节（33-126, 161-172, 174-255）保持不变
        # 不可显示的字节（0-32, 127-160, 173）映射到 256+ 的 Unicode 码点
        bs = (
            list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def encode(self, text: str) -> list[int]:
        """将字符串编码为 token ID 列表。"""
        if self.special_tokens:
            parts = regex.split(self.special_pattern, text)
        else:
            parts = [text]
        indices = []
        for part in parts:
            if part in self.special_tokens:
                indices.append(self.vocab_reverse[part])
            else:
                for word in regex.findall(self.gpt2_pat, part):
                    token_chars = [
                        self.bytes2unicode.get(b, chr(b))
                        for b in word.encode("utf-8")
                    ]
                    token_chars = self.apply_merges(token_chars)
                    indices.extend(self.vocab_reverse[c] for c in token_chars)
        return indices

    def apply_merges_old(self, token_chars: list[str]) -> list[str]:
        """根据 merges 列表合并 token chars。"""
        for k1, k2 in self.merges:
            i = 0
            while i < len(token_chars) - 1:
                if token_chars[i] == k1 and token_chars[i + 1] == k2:
                    token_chars[i : i + 2] = [k1 + k2]
                i += 1
        return token_chars

    def apply_merges(self, token_chars: list[str]) -> list[str]:
        """根据 merges 列表合并 token chars。"""
        word = list(token_chars)
        while True:
            pairs = list(zip(word[:-1], word[1:]))
            if not pairs:
                break
            ranked = [
                (self.merge_ranks[p], p) for p in set(pairs) if p in self.merge_ranks
            ]
            if not ranked:
                break
            _, best_pair = min(ranked, key=lambda x: x[0])
            new_word = []
            i = 0
            while i < len(word):
                j = i
                while j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                    j += 1
                new_word.append("".join(word[i : j + 1]))
                i = j + 1
            word = new_word
        return word

    def decode(self, ids: list[int]) -> str:
        """将 token ID 列表解码为字符串。"""
        for id in ids:
            if id not in self.vocab:
                raise ValueError(f"Token ID {id} not in vocab")
        token_chars = "".join(self.vocab[id] for id in ids)
        bytes_list = [
            self.bytes2unicode_reverse.get(c, ord(c)) for c in token_chars
        ]
        return bytes(bytes_list).decode("utf-8", errors="replace")

    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        """流式编码，内存友好版本。"""
        full_text = "".join(text)
        for i in self.encode(full_text):
            yield i


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    """创建 BPE tokenizer 实例。"""
    return Tokenizer(vocab, merges, special_tokens)
