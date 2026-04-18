"""Part 1 基础测试脚本"""
import json
import sys
sys.path.insert(0, "scripts")
from tokenizer import Tokenizer

FIXTURES = "homework/assignment1/tests/fixtures"
EOT = "<|endoftext|>"

def gpt2_bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def load_tokenizer(special_tokens=None):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(f"{FIXTURES}/gpt2_vocab.json") as f:
        gpt2_vocab = json.load(f)
    with open(f"{FIXTURES}/gpt2_merges.txt") as f:
        gpt2_bpe_merges = []
        for line in f:
            cleaned = line.rstrip()
            if cleaned and len(cleaned.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned.split(" ")))
    vocab = {
        idx: bytes([gpt2_byte_decoder[t] for t in item])
        for item, idx in gpt2_vocab.items()
    }
    if special_tokens:
        for st in special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in set(vocab.values()):
                vocab[len(vocab)] = st_bytes
    merges = [
        (
            bytes([gpt2_byte_decoder[t] for t in m1]),
            bytes([gpt2_byte_decoder[t] for t in m2]),
        )
        for m1, m2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)

# Test 1
print("=== Test 1: empty string ===")
tok = load_tokenizer()
assert tok.encode("") == [] and tok.decode([]) == ""
print("PASS")

# Test 2
print("\n=== Test 2: single char roundtrip ===")
for s in ["s", "a", "z"]:
    assert tok.decode(tok.encode(s)) == s
print("PASS")

# Test 3
print("\n=== Test 3: ASCII roundtrip ===")
text = "Hello, how are you?"
ids = tok.encode(text)
assert tok.decode(ids) == text
print(f"PASS: ids={ids}")
print(f"  tokens: {[tok.decode([i]) for i in ids]}")

# Test 4
print("\n=== Test 4: Unicode roundtrip ===")
for text in ["\U0001f643", "H\xc3\xa9ll\xc3\xb2 h\xc3\xb4w are \xc3\xbc? \U0001f643"]:
    pass  # just define
for text in ["🙃", "Héllò hôw are ü? 🙃"]:
    ids = tok.encode(text)
    assert tok.decode(ids) == text, f"Failed: got '{tok.decode(ids)}'"
print("PASS")

# Test 5
print("\n=== Test 5: tiktoken match ===")
import tiktoken
ref = tiktoken.get_encoding("gpt2")
for text in ["", "s", "🙃", "Hello, how are you?", "Héllò hôw are ü? 🙃"]:
    my_ids = tok.encode(text)
    ref_ids = ref.encode(text)
    assert my_ids == ref_ids, f"Mismatch for {repr(text)}:\n  mine={my_ids}\n  ref ={ref_ids}"
print("PASS: all match tiktoken")

# Test 6
print("\n=== Test 6: special token roundtrip ===")
tok_sp = load_tokenizer(special_tokens=[EOT])
text = "Hello, how are you?"
assert tok_sp.decode(tok_sp.encode(text)) == text
print("PASS")

# Test 7
print("\n=== Test 7: special token in text ===")
text = "Héllò hôw " + EOT + " how are ü? 🙃"
ids = tok_sp.encode(text)
assert tok_sp.decode(ids) == text
tokens = [tok_sp.decode([i]) for i in ids]
assert tokens.count(EOT) == 1
print(f"PASS: {EOT} preserved")

# Test 8
print("\n=== Test 8: tiktoken special token match ===")
ref_ids = ref.encode(text, allowed_special={EOT})
my_ids = tok_sp.encode(text)
assert my_ids == ref_ids, f"Mismatch:\n  mine={my_ids}\n  ref ={ref_ids}"
print("PASS")

# Test 9
print("\n=== Test 9: encode_iterable roundtrip ===")
text = "Hello, how are you?"
assert tok.decode(list(tok.encode_iterable(iter(text)))) == text
print("PASS")

# Test 10
print("\n=== Test 10: file encode_iterable roundtrip ===")
with open(f"{FIXTURES}/tinystories_sample.txt") as f:
    all_ids = list(tok.encode_iterable(f))
with open(f"{FIXTURES}/tinystories_sample.txt") as f:
    corpus = f.read()
assert tok.decode(all_ids) == corpus
print("PASS")

print("\n" + "="*40)
print("ALL TESTS PASSED!")
