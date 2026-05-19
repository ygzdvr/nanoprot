"""Tests for the BERT-style MLM masking in :mod:`nanoprot.data.mlm`.

We test the pure masking function (which is the only piece that does not
need a real parquet dataset on disk). The end-to-end MLM loader is covered
implicitly by the v0.3 training-loop integration test.
"""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from nanoprot.data.mlm import _collect_special_token_ids, _mlm_mask  # noqa: E402
from nanoprot.tokenizers.esm2 import (  # noqa: E402
    ID_CLS,
    ID_EOS,
    ID_MASK,
    ID_PAD,
    Esm2Tokenizer,
)


# ---------------------------------------------------------------------------
# Special-token collection
# ---------------------------------------------------------------------------

class TestSpecialTokenCollection:
    def test_collects_esm2_specials(self) -> None:
        special = _collect_special_token_ids(Esm2Tokenizer())
        # All four ESM-2 special tokens we know about must be collected.
        assert ID_CLS in special
        assert ID_PAD in special
        assert ID_EOS in special
        assert ID_MASK in special

    def test_tokenizer_without_accessors_returns_empty(self) -> None:
        class _Bare:
            pass
        assert _collect_special_token_ids(_Bare()) == set()


# ---------------------------------------------------------------------------
# Masking statistics
# ---------------------------------------------------------------------------

class TestMlmMask:
    def test_targets_are_minus_100_outside_selected_positions(self) -> None:
        gen = torch.Generator().manual_seed(0)
        inputs = torch.randint(4, 33, (4, 32), dtype=torch.long)  # avoid special ids
        masked, targets = _mlm_mask(
            inputs, mask_token_id=ID_MASK, vocab_size=33,
            mlm_probability=0.15, special_ids=set(), generator=gen,
        )
        # Wherever target == -100, the input was NOT selected for masking
        # (so masked == inputs at those positions).
        unselected = targets == -100
        assert torch.equal(masked[unselected], inputs[unselected])

    def test_targets_match_inputs_at_selected_positions(self) -> None:
        gen = torch.Generator().manual_seed(0)
        inputs = torch.randint(4, 33, (4, 32), dtype=torch.long)
        masked, targets = _mlm_mask(
            inputs, mask_token_id=ID_MASK, vocab_size=33,
            mlm_probability=0.15, special_ids=set(), generator=gen,
        )
        selected = targets != -100
        # Targets hold the ORIGINAL ids at selected positions, even when the
        # input was replaced with <mask> or a random token.
        assert torch.equal(targets[selected], inputs[selected])

    def test_selection_fraction_close_to_probability(self) -> None:
        """Over a large enough batch, ~15% of tokens should be selected."""
        gen = torch.Generator().manual_seed(0)
        inputs = torch.randint(4, 33, (16, 1024), dtype=torch.long)
        _, targets = _mlm_mask(
            inputs, mask_token_id=ID_MASK, vocab_size=33,
            mlm_probability=0.15, special_ids=set(), generator=gen,
        )
        frac = float((targets != -100).float().mean())
        assert 0.13 < frac < 0.17, f"selection fraction {frac:.3f} far from 0.15"

    def test_mask_replacement_fraction_close_to_80_percent(self) -> None:
        """Of the selected tokens, ~80% should be replaced with <mask>."""
        gen = torch.Generator().manual_seed(0)
        # Use a vocab that doesn't contain ID_MASK as a candidate input
        # to make the test unambiguous.
        inputs = torch.randint(4, 32, (16, 1024), dtype=torch.long)
        masked, targets = _mlm_mask(
            inputs, mask_token_id=ID_MASK, vocab_size=32,
            mlm_probability=0.15, special_ids=set(), generator=gen,
        )
        selected = targets != -100
        replaced_with_mask = (masked == ID_MASK) & selected
        frac = float(replaced_with_mask.sum()) / max(1, int(selected.sum()))
        assert 0.75 < frac < 0.85, f"mask replacement fraction {frac:.3f} far from 0.80"

    def test_special_tokens_never_masked(self) -> None:
        """Tokens listed in ``special_ids`` must not appear in the target set."""
        gen = torch.Generator().manual_seed(0)
        # Insert specials all over the batch.
        inputs = torch.randint(4, 33, (8, 128), dtype=torch.long)
        inputs[:, 0] = ID_CLS                # BOS at every row's start
        inputs[:, ::16] = ID_PAD             # pads every 16 positions
        specials = {ID_CLS, ID_PAD, ID_EOS, ID_MASK}

        _, targets = _mlm_mask(
            inputs, mask_token_id=ID_MASK, vocab_size=33,
            mlm_probability=0.5,             # high probability -> stress test
            special_ids=specials, generator=gen,
        )
        selected = targets != -100
        # Every selected position must have had a NON-special input token.
        for sid in specials:
            assert not (selected & (inputs == sid)).any(), f"special token {sid} got selected"

    def test_deterministic_when_seeded(self) -> None:
        """Same generator state -> identical masked output (regression test
        for the seed plumbing in the MLM loader)."""
        inputs = torch.randint(4, 33, (4, 64), dtype=torch.long)
        masked1, tgt1 = _mlm_mask(
            inputs, mask_token_id=ID_MASK, vocab_size=33,
            mlm_probability=0.15, special_ids=set(),
            generator=torch.Generator().manual_seed(42),
        )
        masked2, tgt2 = _mlm_mask(
            inputs, mask_token_id=ID_MASK, vocab_size=33,
            mlm_probability=0.15, special_ids=set(),
            generator=torch.Generator().manual_seed(42),
        )
        assert torch.equal(masked1, masked2)
        assert torch.equal(tgt1, tgt2)
