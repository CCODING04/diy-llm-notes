"""Part 7 完整 Transformer 语言模型测试"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import numpy as np
from model_components import TransformerLM
from training import AdamW, train


def test_transformer_lm():
    """TransformerLM: 形状正确 + 参数可训练 + 前向传播"""
    vocab_size = 100
    context_length = 16
    d_model = 32
    num_layers = 2
    num_heads = 4

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads)

    # 检查内部结构
    assert hasattr(model, 'token_embedding')
    assert hasattr(model, 'layers')
    assert hasattr(model, 'norm')
    assert hasattr(model, 'output')
    assert len(model.layers) == num_layers

    # 前向传播
    token_ids = torch.randint(0, vocab_size, (2, context_length))
    logits = model(token_ids)
    assert logits.shape == (2, context_length, vocab_size)

    # 梯度可传播
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

    print("  [PASS] TransformerLM")


def test_transformer_lm_different_sizes():
    """不同配置的 TransformerLM"""
    configs = [
        (50, 8, 16, 1, 2),    # 小模型
        (200, 32, 64, 3, 8),  # 中等模型
    ]
    for vocab_size, ctx, d_model, n_layers, n_heads in configs:
        model = TransformerLM(vocab_size, ctx, d_model, n_layers, n_heads)
        x = torch.randint(0, vocab_size, (1, ctx))
        out = model(x)
        assert out.shape == (1, ctx, vocab_size), f"配置 {(vocab_size, ctx, d_model, n_layers, n_heads)} 失败"

    print("  [PASS] TransformerLM different sizes")


def test_train_loop():
    """训练循环: loss 应该下降"""
    model = TransformerLM(vocab_size=64, context_length=16, d_model=32, num_layers=1, num_heads=4)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    data = np.random.randint(0, 64, 500)

    # 记录初始 loss
    model.eval()
    with torch.no_grad():
        x_init = torch.randint(0, 64, (4, 16))
        y_init = torch.randint(0, 64, (4, 16))
        init_logits = model(x_init)
        init_loss = torch.nn.functional.cross_entropy(
            init_logits.view(-1, 64), y_init.view(-1)
        ).item()

    # 训练
    train(model, data, optimizer, batch_size=4, context_length=16,
          device='cpu', max_iters=50, log_interval=25)

    # 训练后 loss 应该更低
    model.eval()
    with torch.no_grad():
        final_logits = model(x_init)
        final_loss = torch.nn.functional.cross_entropy(
            final_logits.view(-1, 64), y_init.view(-1)
        ).item()

    # loss 应该下降
    assert final_loss < init_loss, f"loss 未下降: {init_loss:.4f} → {final_loss:.4f}"

    print(f"  [PASS] train loop (loss: {init_loss:.4f} → {final_loss:.4f})")


if __name__ == "__main__":
    print("Part 7 完整 Transformer 语言模型测试")
    print("=" * 45)
    test_transformer_lm()
    test_transformer_lm_different_sizes()
    test_train_loop()
    print("=" * 45)
    print("全部测试通过!")
