"""
==============================================================================
MODULE 18 — GRAMMAR-CONSTRAINED & FINITE-STATE MACHINE (FSM) STRUCTURED DECODING
==============================================================================
OBJECTIVE:
    Guarantee 100% syntactically valid JSON and tool-call DSL generation from
    compact (<2B) quantized SLMs on the first pass with zero retry loops:
    1. `FiniteStateGrammarIndex`: Pre-indexes a Deterministic Finite Automaton
       (DFA / FSM) over the vocabulary V so each automaton state q in Q maps in
       O(1) time to its allowed token subset A(q) subset_eq V (Willard & Louf,
       Outlines 2023, arXiv:2307.09702; Dong et al., XGrammar 2024, arXiv:2411.15100).
    2. `apply_fsm_logit_mask`: Zero-overhead additive logit mask setting invalid
       vocabulary transitions to -inf while preserving the exact conditional
       distribution P(v | x, v in A(q_t)).
    3. `JSONSchemaFSMDecoder`: Compiles a typed JSON schema into a deterministic
       state machine and decodes guaranteed-valid JSON payloads from arbitrary
       SLM logit streams.

RUN COMMAND:
    python src/module_18_constrained_decoding/fsm_guided_decoding.py
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstrainedDecodingReport:
    """Telemetry from FSM-guided structured decoding."""

    generated_text: str
    parsed_json: Dict[str, object]
    steps_executed: int
    syntax_valid: bool
    schema_compliant: bool
    avg_allowed_vocab_fraction: float


def apply_fsm_logit_mask(
    logits: torch.Tensor,
    allowed_token_ids: Sequence[int],
) -> torch.Tensor:
    """
    Apply a deterministic FSM vocabulary mask to raw logits z in R^(...|V|):
        z'_v = z_v       if v in allowed_token_ids
        z'_v = -inf      otherwise

    Preserves the exact conditional probability distribution over valid tokens:
        softmax(z')_v = P(v | x) / sum_{u in A(q)} P(u | x)   for v in A(q).
    """
    if len(allowed_token_ids) == 0:
        raise ValueError("allowed_token_ids cannot be empty for an active non-terminal FSM state.")
    vocab_size = logits.shape[-1]
    for tid in allowed_token_ids:
        if tid < 0 or tid >= vocab_size:
            raise ValueError(f"Token ID {tid} out of bounds for vocab_size={vocab_size}.")

    mask = torch.full_like(logits, float("-inf"))
    idx_tensor = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=logits.device)
    mask.index_copy_(-1, idx_tensor, logits.index_select(-1, idx_tensor))
    return mask


class FiniteStateGrammarIndex:
    """
    Pre-compiled Deterministic Finite Automaton (DFA) vocabulary index
    (Willard & Louf, 2023, arXiv:2307.09702):
        - `transitions[state][token_id] -> next_state`
        - `allowed_tokens(state) -> Tuple[int, ...]` in O(1) lookup time.
    """

    def __init__(
        self,
        vocab: Dict[int, str],
        transitions: Dict[int, Dict[int, int]],
        initial_state: int = 0,
        accepting_states: Sequence[int] = (),
    ) -> None:
        self.vocab = dict(vocab)
        self.transitions = {s: dict(tmap) for s, tmap in transitions.items()}
        self.initial_state = initial_state
        self.accepting_states = set(accepting_states)

        # Precompute O(1) allowed token tuple per state
        self._state_allowed_cache: Dict[int, Tuple[int, ...]] = {
            state: tuple(sorted(tmap.keys())) for state, tmap in self.transitions.items()
        }

    def allowed_tokens(self, state: int) -> Tuple[int, ...]:
        if state not in self._state_allowed_cache:
            return ()
        return self._state_allowed_cache[state]

    def next_state(self, state: int, token_id: int) -> int:
        if state not in self.transitions or token_id not in self.transitions[state]:
            raise ValueError(f"Invalid FSM transition from state={state} on token_id={token_id}.")
        return self.transitions[state][token_id]

    def is_accepting(self, state: int) -> bool:
        return state in self.accepting_states


class JSONSchemaFSMDecoder:
    """
    Compiles a flat typed JSON schema (mapping field_name -> 'string' | 'number' | 'boolean')
    into a `FiniteStateGrammarIndex` over a subword/token vocabulary and executes
    guaranteed-valid greedy or sampling decoding from arbitrary SLM logits.
    """

    def __init__(self, field_types: Dict[str, str]) -> None:
        if not field_types:
            raise ValueError("field_types schema cannot be empty.")
        self.field_types = dict(field_types)
        self.vocab, self.token_to_id = self._build_domain_vocabulary(self.field_types)
        self.fsm = self._compile_schema_fsm(self.field_types)

    @staticmethod
    def _build_domain_vocabulary(field_types: Dict[str, str]) -> Tuple[Dict[int, str], Dict[str, int]]:
        tokens: List[str] = [
            "{",
            "}",
            ":",
            ",",
            '"true"',
            "true",
            "false",
            '"ok"',
            '"slm_edge"',
            '"high"',
            "42",
            "100",
            "0.95",
            "<invalid_ garbage>",
            "syntax_error_token",
            "null_oops",
        ]
        for key in field_types:
            tok = f'"{key}"'
            if tok not in tokens:
                tokens.append(tok)
        vocab = {idx: tok for idx, tok in enumerate(tokens)}
        token_to_id = {tok: idx for idx, tok in vocab.items()}
        return vocab, token_to_id

    def _compile_schema_fsm(self, field_types: Dict[str, str]) -> FiniteStateGrammarIndex:
        transitions: Dict[int, Dict[int, int]] = {}
        state = 0

        # State 0 -> '{' -> State 1
        transitions[state] = {self.token_to_id["{"]: state + 1}
        state += 1

        fields = list(field_types.items())
        for idx, (fname, ftype) in enumerate(fields):
            # Expect '"field_name"'
            key_tid = self.token_to_id[f'"{fname}"']
            transitions[state] = {key_tid: state + 1}
            state += 1

            # Expect ':'
            colon_tid = self.token_to_id[":"]
            transitions[state] = {colon_tid: state + 1}
            state += 1

            # Expect typed value token(s)
            if ftype == "string":
                val_candidates = [self.token_to_id['"ok"'], self.token_to_id['"slm_edge"'], self.token_to_id['"high"']]
            elif ftype == "number":
                val_candidates = [self.token_to_id["42"], self.token_to_id["100"], self.token_to_id["0.95"]]
            elif ftype == "boolean":
                val_candidates = [self.token_to_id["true"], self.token_to_id["false"]]
            else:
                raise ValueError(f"Unsupported schema field type: {ftype}")

            transitions[state] = {v_tid: state + 1 for v_tid in val_candidates}
            state += 1

            # Expect ',' between fields or '}' after final field
            if idx < len(fields) - 1:
                comma_tid = self.token_to_id[","]
                transitions[state] = {comma_tid: state + 1}
                state += 1
            else:
                close_tid = self.token_to_id["}"]
                transitions[state] = {close_tid: state + 1}
                state += 1

        accepting_state = state
        return FiniteStateGrammarIndex(
            vocab=self.vocab,
            transitions=transitions,
            initial_state=0,
            accepting_states=[accepting_state],
        )

    def decode_with_fsm(
        self,
        raw_step_logits: torch.Tensor,
    ) -> ConstrainedDecodingReport:
        """
        Decode a structured JSON object using FSM logit masking at every step.

        Args:
            raw_step_logits: Tensor of shape (max_steps, vocab_size) representing
                             unconstrained SLM logits (which may heavily favor invalid tokens).
        """
        state = self.fsm.initial_state
        vocab_size = len(self.vocab)
        emitted_Pieces: List[str] = []
        allowed_fractions: List[float] = []
        step = 0

        while not self.fsm.is_accepting(state):
            if step >= raw_step_logits.shape[0]:
                raise RuntimeError("Exceeded raw_step_logits length before reaching accepting FSM state.")
            allowed = self.fsm.allowed_tokens(state)
            allowed_fractions.append(len(allowed) / float(vocab_size))

            masked_logits = apply_fsm_logit_mask(raw_step_logits[step], allowed)
            probs = F.softmax(masked_logits, dim=-1)
            chosen_tid = int(torch.argmax(probs).item())

            emitted_Pieces.append(self.vocab[chosen_tid])
            state = self.fsm.next_state(state, chosen_tid)
            step += 1

        generated = "".join(emitted_Pieces)
        parsed = json.loads(generated)
        schema_ok = set(parsed.keys()) == set(self.field_types.keys())

        return ConstrainedDecodingReport(
            generated_text=generated,
            parsed_json=parsed,
            steps_executed=step,
            syntax_valid=True,
            schema_compliant=schema_ok,
            avg_allowed_vocab_fraction=round(sum(allowed_fractions) / max(len(allowed_fractions), 1), 4),
        )


def main() -> None:
    torch.manual_seed(42)
    schema = {"route": "string", "confidence": "number", "pii_safe": "boolean"}
    decoder = JSONSchemaFSMDecoder(schema)

    # Deliberately bias unconstrained logits toward invalid syntax garbage tokens
    vocab_size = len(decoder.vocab)
    adversarial_logits = torch.randn(32, vocab_size)
    adversarial_logits[:, decoder.token_to_id["<invalid_ garbage>"]] += 50.0
    adversarial_logits[:, decoder.token_to_id["syntax_error_token"]] += 40.0

    # Without FSM: argmax picks '<invalid_ garbage>' at every step (0% valid JSON)
    unconstrained_ids = torch.argmax(adversarial_logits[:13], dim=-1).tolist()
    unconstrained_text = "".join(decoder.vocab[i] for i in unconstrained_ids)

    # With FSM: 100% valid JSON and 100% schema compliance on the first attempt
    report = decoder.decode_with_fsm(adversarial_logits)

    logger.info("=" * 72)
    logger.info("MODULE 18: GRAMMAR-CONSTRAINED & FSM STRUCTURED DECODING (Outlines/XGrammar)")
    logger.info("=" * 72)
    logger.info(f"[Unconstrained Decode] Output: {unconstrained_text[:55]}... (JSON Valid=False)")
    logger.info(
        f"[FSM-Guided Decode]    Output: {report.generated_text} "
        f"| JSON Valid={report.syntax_valid} | Schema Compliant={report.schema_compliant} "
        f"| Avg Allowed Vocab={report.avg_allowed_vocab_fraction:.1%}"
    )


if __name__ == "__main__":
    main()
