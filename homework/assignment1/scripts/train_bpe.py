"""Assignment 1 - BPE 训练实现

从原始文本语料训练 BPE tokenizer，输出 vocab 和 merges。
参考：https://github.com/stanford-cs336/assignment1-basics
"""

import os
from collections import Counter

import regex


def bytes_to_unicode():
    """GPT-2 字节到 Unicode 映射。"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word):
    """获取词中所有相邻 pair。"""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_pair(word, pair):
    """合并词中的指定 pair，返回新的 tuple。"""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(pair[0] + pair[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def run_train_bpe_set(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从语料训练 BPE tokenizer。

    Args:
        input_path: 训练语料路径
        vocab_size: 目标词表大小（含 special tokens）
        special_tokens: 特殊 token 列表

    Returns:
        (vocab, merges) 词表和合并规则
    """
    # TODO: 实现完整的 BPE 训练流程
    vocab = {}
    merges = []

    # 读取文件内容
    with open(input_path, "r") as f:
        text = f.read()

    # 先按照 special token 分割
    valid_specials = [t for t in special_tokens if t != ""]
    if valid_specials:
        special_pattern = "(" + "|".join(map(regex.escape, valid_specials)) + ")"
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freqs = Counter()
    for part in parts:
        words = regex.findall(GPT2_PAT, part)
        for word in words:
            word_freqs[word] += 1

    bytes_unicode = bytes_to_unicode()
    vocab = {}
    for word, freq in word_freqs.items():
        word_bytes = word.encode("utf-8")
        word_tokens = tuple(bytes_unicode[b] for b in word_bytes)
        vocab[word_tokens] = freq

    num_merges = vocab_size - len(special_tokens) - 256

    for _ in range(num_merges):

        pairs = Counter()
        for word_tokens, freq in vocab.items():
            # for i in range(len(word_tokens) - 1):
            #     pairs[(word_tokens[i], word_tokens[i+1])] += freq
            for piece in zip(word_tokens[:-1], word_tokens[1:]):
                pairs[piece] += freq
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)

        new_vocab = {}
        for word_tokens, freq in vocab.items():
            new_vocab[merge_pair(word_tokens, best)] = freq

        vocab = new_vocab

    unicode_byte = {v: k for k, v in bytes_unicode.items()}
    final_vocab = {}
    for i, sp in enumerate(special_tokens):
        final_vocab[i] = sp.encode("utf-8")

    offset = len(special_tokens)
    for i in range(256):
        b = i + offset
        final_vocab[b] = bytes([i])

    for i, (s1, s2) in enumerate(merges):
        merged = s1 + s2
        final_vocab[i + offset + 256] = bytes(unicode_byte[c] for c in merged)

    final_merges = []
    for s1, s2 in merges:
        final_merges.append(
            (
                bytes(unicode_byte[c] for c in s1),
                bytes(unicode_byte[c] for c in s2),
            )
        )

    return (final_vocab, final_merges)


def run_train_bpe_v1(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从语料训练 BPE tokenizer。

    Args:
        input_path: 训练语料路径
        vocab_size: 目标词表大小（含 special tokens）
        special_tokens: 特殊 token 列表

    Returns:
        (vocab, merges) 词表和合并规则
    """
    # TODO: 实现完整的 BPE 训练流程

    merges = []

    # 读取文件内容
    with open(input_path, "r") as f:
        text = f.read()

    # 先按照 special token 分割
    valid_specials = [t for t in special_tokens if t != ""]
    if valid_specials:
        special_pattern = "(" + "|".join(map(regex.escape, valid_specials)) + ")"
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    GPT2_PAT = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{L}+|?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    bytes_unicode = bytes_to_unicode()

    words = []
    for part in parts:
        words_part = regex.findall(GPT2_PAT, part)
        for word in words_part:
            words.append(list(bytes_unicode[b] for b in word.encode("utf-8")))

    num_merges = vocab_size - 256 - len(special_tokens)

    for _ in range(num_merges):
        stats = Counter()
        for word in words:
            for pair in zip(word[:-1], word[1:]):
                stats[pair] += 1
        if not stats:
            break

        best = max(stats, key=lambda x: (stats[x], x))
        merges.append(best)
        for word in words:
            i = 0
            while i < len(word) - 1:
                if word[i] == best[0] and word[i + 1] == best[1]:
                    word[i : i + 2] = [best[0] + best[1]]
                i += 1

    unicode_byte = {v: k for k, v in bytes_unicode.items()}
    final_vocab = {}
    for i, sp in enumerate(special_tokens):
        final_vocab[i] = sp.encode("utf-8")

    offset = len(special_tokens)
    for i in range(256):
        b = i + offset
        final_vocab[b] = bytes([i])

    for i, (s1, s2) in enumerate(merges):
        merged = s1 + s2
        final_vocab[i + offset + 256] = bytes(unicode_byte[c] for c in merged)

    final_merges = []
    for s1, s2 in merges:
        final_merges.append(
            (
                bytes(unicode_byte[c] for c in s1),
                bytes(unicode_byte[c] for c in s2),
            )
        )
    return (final_vocab, final_merges)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "r") as f:
        text = f.read()

    GPT2_PAT = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    valid_special_tokens = [t for t in special_tokens if t != ""]
    if valid_special_tokens:
        special_pattern = "(" + "|".join(map(regex.escape, valid_special_tokens)) + ")"
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    bytes_unicode = bytes_to_unicode()
    word_freqs = {}
    for part in parts:
        for word in regex.findall(GPT2_PAT, part):
            key = tuple(bytes_unicode[b] for b in word.encode("utf-8"))
            word_freqs[key] = word_freqs.get(key, 0) + 1

    sort_key_cache = {}

    unicode_byte = {v:k for k, v in bytes_unicode.items()}
    def get_sort_key(t):
        if t not in sort_key_cache:
            sort_key_cache[t] = bytes(unicode_byte[c] for c in t)
        return sort_key_cache[t]

    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        pair_counts = {}
        for word, freq in word_freqs.items():
            for pair in zip(word[:-1], word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        if not pair_counts:
            break
        max_count = max(pair_counts.values())
        best = max(
            (p for p in pair_counts if pair_counts[p] == max_count),
            key=lambda x: (get_sort_key(x[0]), get_sort_key(x[1])),
        )
        merges.append(best)
        merged_token = best[0] + best[1]
        if merged_token not in sort_key_cache:
            sort_key_cache[merged_token] = get_sort_key(best[0]) + get_sort_key(best[1])
        new_freqs = {}
        b0, b1 = best
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == b0 and word[i + 1] == b1:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        word_freqs = new_freqs

    final_vocab = {}
    for i, sp in enumerate(special_tokens):
        final_vocab[i] = sp.encode("utf-8")

    offset = len(special_tokens)
    for i in range(256):
        b = i + offset
        final_vocab[b] = bytes([i])

    for i, (s1, s2) in enumerate(merges):
        merged = s1 + s2
        final_vocab[i + offset + 256] = bytes(unicode_byte[c] for c in merged)

    final_merges = []
    for s1, s2 in merges:
        final_merges.append(
            (
                bytes(unicode_byte[c] for c in s1),
                bytes(unicode_byte[c] for c in s2),
            )
        )
    return (final_vocab, final_merges)
