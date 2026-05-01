"""W4 验收测试：完整 mini-LLaMA"""

import torch
import pytest

from mini_llama.model import MiniLlama, ModelConfig
from mini_llama.norm.rms_norm import RMSNorm


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=128)
        x = torch.randn(2, 16, 128)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalizes_to_unit_rms(self):
        """RMSNorm(x) 的 RMS 应该约为 1（weight 初始化为 1 时）"""
        norm = RMSNorm(dim=128)
        x = torch.randn(2, 16, 128) * 100
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


class TestMiniLlamaSmoke:
    """W4 必过：模型能 forward + 输出形状正确"""

    @pytest.mark.parametrize("attn_type", ["mha", "gqa", "mla"])
    def test_forward(self, attn_type):
        config = ModelConfig(
            vocab_size=100,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            attention_type=attn_type,
        )
        model = MiniLlama(config)
        tokens = torch.randint(0, 100, (2, 16))
        logits = model(tokens)
        assert logits.shape == (2, 16, 100)

    def test_can_compute_loss(self):
        """能算 loss = 能用于训练"""
        config = ModelConfig(vocab_size=100, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
        model = MiniLlama(config)
        tokens = torch.randint(0, 100, (2, 16))
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 100),
            tokens[:, 1:].reshape(-1),
        )
        assert torch.isfinite(loss).item()
        loss.backward()  # gradient flows
