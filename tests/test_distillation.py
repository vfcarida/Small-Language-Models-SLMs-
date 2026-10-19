from unittest.mock import MagicMock, patch
import pytest
import requests
import torch
from src.module_03_distillation.gkd_pipeline import (
    VLLMTeacherOracle,
    compute_distillation_loss,
    explain_kl_divergence,
)


def test_distillation_loss_zero_when_identical():
    logits = torch.tensor([[[2.0, -1.0, 0.5, 3.0]]])
    for mode in ("forward_kl", "reverse_kl", "jsd"):
        loss = compute_distillation_loss(logits, logits.clone(), temperature=2.0, mode=mode)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_distillation_loss_non_negative_and_jsd_symmetry():
    s_logits = torch.tensor([[[1.5, 0.2, -0.8, 0.0], [0.1, 2.0, -1.0, 0.5]]])
    t_logits = torch.tensor([[[-0.5, 2.5, 1.0, -1.2], [1.2, -0.4, 0.8, 0.1]]])

    fkl = compute_distillation_loss(s_logits, t_logits, temperature=2.0, mode="forward_kl")
    rkl = compute_distillation_loss(s_logits, t_logits, temperature=2.0, mode="reverse_kl")
    jsd_st = compute_distillation_loss(s_logits, t_logits, temperature=2.0, mode="jsd", beta=0.5)
    jsd_ts = compute_distillation_loss(t_logits, s_logits, temperature=2.0, mode="jsd", beta=0.5)

    assert fkl.item() > 0.0
    assert rkl.item() > 0.0
    assert jsd_st.item() > 0.0
    # Symmetric JSD (beta=0.5) must be symmetric in its arguments
    assert torch.isclose(jsd_st, jsd_ts, atol=1e-5)


def test_distillation_loss_masking():
    s_logits = torch.tensor([[[1.0, 2.0], [10.0, -10.0]]])
    t_logits = torch.tensor([[[1.0, 2.0], [-10.0, 10.0]]])
    # Mask out the second token where student and teacher strongly disagree
    mask = torch.tensor([[1.0, 0.0]])
    masked_loss = compute_distillation_loss(s_logits, t_logits, temperature=1.0, mode="forward_kl", mask=mask)
    assert torch.isclose(masked_loss, torch.tensor(0.0), atol=1e-5)


def test_distillation_loss_invalid_inputs():
    logits = torch.randn(2, 4)
    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        compute_distillation_loss(logits, logits, temperature=0.0)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_distillation_loss(logits, torch.randn(2, 5))
    with pytest.raises(ValueError, match="Unsupported distillation mode"):
        compute_distillation_loss(logits, logits, mode="invalid")  # type: ignore[arg-type]


def test_explain_kl_divergence():
    metrics = explain_kl_divergence()
    assert metrics["forward_kl"] > 0
    assert metrics["reverse_kl"] > 0
    assert metrics["jsd"] > 0


@patch("src.module_03_distillation.gkd_pipeline.requests.post")
@patch("src.module_03_distillation.gkd_pipeline.time.sleep")
def test_vllm_teacher_oracle_retries_then_succeeds(mock_sleep, mock_post):
    fail_resp = requests.exceptions.ConnectionError("Service busy")
    ok_resp = MagicMock()
    ok_resp.raise_for_status.return_value = None
    ok_resp.json.return_value = {"choices": [{"logprobs": {"top_logprobs": [{"Paris": -0.1}]}}]}

    mock_post.side_effect = [fail_resp, ok_resp]
    oracle = VLLMTeacherOracle("http://localhost:8000", max_retries=3, api_key="test-key")
    result = oracle.get_logprobs("Capital of France:", " Paris")
    assert "choices" in result
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()
