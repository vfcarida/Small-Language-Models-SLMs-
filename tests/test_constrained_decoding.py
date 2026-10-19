"""
Unit tests for Module 18: Grammar-Constrained & FSM Structured Decoding.
"""

import pytest
import torch
import torch.nn.functional as F

from src.module_18_constrained_decoding.fsm_guided_decoding import (
    FiniteStateGrammarIndex,
    JSONSchemaFSMDecoder,
    apply_fsm_logit_mask,
)


def test_apply_fsm_logit_mask_preserves_relative_probabilities():
    logits = torch.tensor([1.0, 2.0, 5.0, -1.0])
    masked = apply_fsm_logit_mask(logits, allowed_token_ids=[0, 1])

    probs = F.softmax(masked, dim=-1)
    assert probs[2].item() == 0.0
    assert probs[3].item() == 0.0
    # Relative probability ratio exp(2.0 - 1.0) = e must be preserved exactly
    assert torch.allclose(probs[1] / probs[0], torch.exp(torch.tensor(1.0)), atol=1e-5)

    with pytest.raises(ValueError, match="allowed_token_ids cannot be empty"):
        apply_fsm_logit_mask(logits, allowed_token_ids=[])


def test_finite_state_grammar_index_transitions():
    vocab = {0: "{", 1: '"a"', 2: ":", 3: "true", 4: "}"}
    transitions = {0: {0: 1}, 1: {1: 2}, 2: {2: 3}, 3: {3: 4}, 4: {4: 5}}
    fsm = FiniteStateGrammarIndex(vocab, transitions, initial_state=0, accepting_states=[5])

    assert fsm.allowed_tokens(0) == (0,)
    assert fsm.next_state(0, 0) == 1
    assert fsm.is_accepting(5) is True

    with pytest.raises(ValueError, match="Invalid FSM transition"):
        fsm.next_state(0, 4)


def test_json_schema_fsm_decoder_adversarial_logits():
    schema = {"task": "string", "score": "number", "verified": "boolean"}
    decoder = JSONSchemaFSMDecoder(schema)

    # Strongly bias logits toward invalid syntax tokens
    vocab_size = len(decoder.vocab)
    logits = torch.randn(24, vocab_size)
    logits[:, decoder.token_to_id["<invalid_ garbage>"]] += 100.0

    report = decoder.decode_with_fsm(logits)
    assert report.syntax_valid is True
    assert report.schema_compliant is True
    assert set(report.parsed_json.keys()) == {"task", "score", "verified"}
    assert 0.0 < report.avg_allowed_vocab_fraction < 0.5
