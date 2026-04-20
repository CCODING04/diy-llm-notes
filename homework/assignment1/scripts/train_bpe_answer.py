"""Assignment 1 - BPE 训练实现（参考答案）

从原始文本语料训练 BPE tokenizer，输出 vocab 和 merges。
参考：https://github.com/stanford-cs336/assignment1-basics
"""

import os

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


def run_train_bpe(
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
    merges = []

    # 读取文件内容
    with open(input_path, "r") as f:
        text = f.read()

    # 先按照 special token 分割（过滤空字符串）
    valid_specials = [t for t in special_tokens if t != ""]
    if valid_specials:
        special_pattern = "(" + "|".join(map(regex.escape, valid_specials)) + ")"
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    # GPT-2 预分词正则（包含数字匹配 \p{N}+）
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    bytes_unicode = bytes_to_unicode()
    unicode_byte = {v: k for k, v in bytes_unicode.items()}

    # 构建词频表 {word_tuple: freq}
    word_freqs = {}
    for part in parts:
        for word in regex.findall(GPT2_PAT, part):
            key = tuple(bytes_unicode[b] for b in word.encode("utf-8"))
            word_freqs[key] = word_freqs.get(key, 0) + 1

    num_merges = vocab_size - 256 - len(special_tokens)

    # 排序键缓存：Unicode 字符串 → 原始字节（用于 tie-breaking）
    sort_key_cache = {}

    def get_sort_key(token_str):
        if token_str not in sort_key_cache:
            sort_key_cache[token_str] = bytes(unicode_byte[c] for c in token_str)
        return sort_key_cache[token_str]

    for _ in range(num_merges):
        # 统计所有相邻 pair 的频率（加权）
        pair_counts = {}
        for word, freq in word_freqs.items():
            for pair in zip(word[:-1], word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        if not pair_counts:
            break

        # 找频率最高的 pair；频率相同时按原始字节字典序选最大
        max_count = max(pair_counts.values())
        best = max(
            (p for p in pair_counts if pair_counts[p] == max_count),
            key=lambda x: (get_sort_key(x[0]), get_sort_key(x[1])),
        )
        merges.append(best)

        # 缓存合并后的 token 的排序键
        merged_token = best[0] + best[1]
        if merged_token not in sort_key_cache:
            sort_key_cache[merged_token] = get_sort_key(best[0]) + get_sort_key(
                best[1]
            )

        # 合并所有词中的 best pair
        b0, b1 = best
        new_freqs = {}
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

    # 构建最终 vocab
    final_vocab = {}
    for i, sp in enumerate(special_tokens):
        final_vocab[i] = sp.encode("utf-8")

    offset = len(special_tokens)
    for i in range(256):
        final_vocab[i + offset] = bytes([i])

    for i, (s1, s2) in enumerate(merges):
        merged = s1 + s2
        final_vocab[i + offset + 256] = bytes(unicode_byte[c] for c in merged)

    # 构建最终 merges（Unicode → 原始字节）
    final_merges = [
        (
            bytes(unicode_byte[c] for c in s1),
            bytes(unicode_byte[c] for c in s2),
        )
        for s1, s2 in merges
    ]

    return (final_vocab, final_merges)
