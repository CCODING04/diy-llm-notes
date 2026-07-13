"""Part 8 文本生成与端到端验证测试"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import numpy as np
from model_components import TransformerLM, softmax
from training import AdamW, train, evaluate, get_batch


def test_generate():
    """generate: 输出形状 + 温度影响"""
    model = TransformerLM(vocab_size=50, context_length=32, d_model=32, num_layers=1, num_heads=4)
    prompt = torch.randint(0, 50, (1, 5))

    # 基本生成
    output = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    assert output.shape == (1, 15), f"形状错误: {output.shape}"
    # 前 5 个 token 应该和 prompt 一致
    assert torch.equal(output[:, :5], prompt)

    # 贪心生成（temperature→0 应该确定性）
    out1 = model.generate(prompt, max_new_tokens=5, temperature=1e-8)
    out2 = model.generate(prompt, max_new_tokens=5, temperature=1e-8)
    assert torch.equal(out1, out2), "贪心生成应该确定性"

    # batch 生成
    prompt_batch = torch.randint(0, 50, (3, 5))
    out_batch = model.generate(prompt_batch, max_new_tokens=5)
    assert out_batch.shape == (3, 10)

    print("  [PASS] generate")


def test_evaluate():
    """evaluate: 返回 loss 和 perplexity"""
    model = TransformerLM(vocab_size=50, context_length=16, d_model=32, num_layers=1, num_heads=4)
    data = np.random.randint(0, 50, 500)

    result = evaluate(model, data, batch_size=4, context_length=16, device='cpu', num_batches=5)
    assert "loss" in result
    assert "perplexity" in result
    assert result["loss"] > 0
    assert result["perplexity"] > 1.0  # PPL > 1

    print(f"  [PASS] evaluate (loss={result['loss']:.4f}, ppl={result['perplexity']:.2f})")


def test_end_to_end():
    """端到端: 训练后 PPL 应该下降"""
    vocab_size = 50
    model = TransformerLM(vocab_size, context_length=16, d_model=32, num_layers=1, num_heads=4)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    data = np.random.randint(0, vocab_size, 1000)

    # 训练前 PPL
    ppl_before = evaluate(model, data, batch_size=4, context_length=16, device='cpu', num_batches=5)["perplexity"]

    # 训练
    train(model, data, optimizer, batch_size=4, context_length=16,
          device='cpu', max_iters=100, log_interval=50)

    # 训练后 PPL
    ppl_after = evaluate(model, data, batch_size=4, context_length=16, device='cpu', num_batches=5)["perplexity"]

    assert ppl_after < ppl_before, f"PPL 未下降: {ppl_before:.2f} → {ppl_after:.2f}"

    # 生成测试
    prompt = torch.randint(0, vocab_size, (1, 5))
    output = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    assert output.shape == (1, 15)

    print(f"  [PASS] end_to_end (PPL: {ppl_before:.2f} → {ppl_after:.2f})")


if __name__ == "__main__":
    print("Part 8 文本生成与端到端验证测试")
    print("=" * 45)
    test_generate()
    test_evaluate()
    test_end_to_end()
    print("=" * 45)
    print("全部测试通过!")
