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
    # TODO: 实现完整的 BPE 训练流程
    raise NotImplementedError
