"""Part 3 整合测试：BPE 训练 + Tokenizer 端到端验证"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from tokenizer import Tokenizer
from train_bpe import run_train_bpe

FIXTURES = "fixtures"

# ========== 测试 1：BPE 训练 ==========
print("=== Test 1: BPE 训练正确性与速度 ===")
start = time.time()
vocab, merges = run_train_bpe(
    input_path=f"{FIXTURES}/corpus.en",
    vocab_size=500,
    special_tokens=[""],
)
elapsed = time.time() - start
print(f"  Vocab: {len(vocab)}, Merges: {len(merges)}, Time: {elapsed:.2f}s")
assert len(vocab) == 500, f"Expected 500, got {len(vocab)}"
assert len(merges) == 243, f"Expected 243, got {len(merges)}"
assert elapsed < 1.5, f"Too slow: {elapsed:.2f}s > 1.5s"
print("  PASS")

# ========== 测试 2：训练结果用于 Tokenizer（roundtrip）==========
print("\n=== Test 2: 训练结果创建 Tokenizer ===")
tok = Tokenizer(vocab, merges, special_tokens=[""])
test_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, there was a little girl.",
    "She said 'hello' and walked away.",
    "Numbers: 123 and 456!",
]
for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"Roundtrip failed: {text!r} -> {decoded!r}"
print(f"  PASS: {len(test_texts)} texts roundtrip correctly")

# ========== 测试 3：Special token roundtrip ==========
print("\n=== Test 3: Special token roundtrip ===")
text_with_sp = "Hello world" + "" + " how are you?"
ids = tok.encode(text_with_sp)
decoded = tok.decode(ids)
assert decoded == text_with_sp, f"Roundtrip failed: {decoded!r}"
tokens = [tok.decode([i]) for i in ids]
assert "" in tokens, "Special token not preserved"
print(f"  PASS: special token preserved")

# ========== 测试 4：在大文件上训练 ==========
print("\n=== Test 4: 小文件训练 ===")
vocab2, merges2 = run_train_bpe(
    input_path=f"{FIXTURES}/tinystories_sample.txt",
    vocab_size=300,
    special_tokens=[""],
)
print(f"  Vocab: {len(vocab2)}, Merges: {len(merges2)}")
tok2 = Tokenizer(vocab2, merges2, special_tokens=[""])
for text in ["Once upon a time.", "She liked to play.", "The end"]:
    ids = tok2.encode(text)
    decoded = tok2.decode(ids)
    assert decoded == text, f"Roundtrip failed: {text!r} -> {decoded!r}"
print("  PASS: roundtrip on small corpus")

# ========== 测试 5：无 special token ==========
print("\n=== Test 5: 无 special token ===")
vocab3, merges3 = run_train_bpe(
    input_path=f"{FIXTURES}/corpus.en",
    vocab_size=500,
    special_tokens=[""],
)
tok3 = Tokenizer(vocab3, merges3)
text = "Hello world!"
ids = tok3.encode(text)
decoded = tok3.decode(ids)
assert decoded == text, f"Roundtrip failed: {decoded!r}"
print("  PASS")

print("\n" + "=" * 40)
print("ALL INTEGRATION TESTS PASSED!")
