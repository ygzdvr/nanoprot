"""B-GEN genome tokenizer + harness integration — regression guard for the additive edits
(tokenizer/dataset Literals, build_tokenizer dispatch, the new GenomeTokenizer). Mirrors
test_esm2_tokenizer.py; formalizes the CPU validations done inline when building B-GEN."""
from __future__ import annotations

from nanoprot.tokenizers.genome import GenomeTokenizer, get_genome_tokenizer, GENOME_TOKENS


def test_vocab_and_special_ids():
    t = get_genome_tokenizer()
    assert t.get_vocab_size() == 10
    assert GENOME_TOKENS[4:9] == ["A", "C", "G", "T", "N"]
    assert [t.get_bos_token_id(), t.get_pad_token_id(), t.get_eos_token_id(),
            t.get_mask_token_id()] == [0, 1, 2, 9]


def test_encode_decode_roundtrip():
    t = get_genome_tokenizer()
    assert t.encode("ACGTN")[0] == [4, 5, 6, 7, 8]
    assert t.decode(t.encode("ACGTACGTNN")[0]) == "ACGTACGTNN"
    assert t.encode("ACGX")[0] == [4, 5, 6, 3]          # unknown char -> <unk> (id 3)
    assert t.encode("ACG", prepend=t.get_bos_token_id())[0] == [0, 4, 5, 6]
    assert t.encode(["AC", "GT"]) == [[4, 5], [6, 7]]   # batched
    assert isinstance(GenomeTokenizer(), GenomeTokenizer)


def test_singleton_cached():
    assert get_genome_tokenizer() is get_genome_tokenizer()


def test_build_tokenizer_dispatch_and_config():
    """A generated genome config validates and dispatches to the GenomeTokenizer."""
    from scripts.gen_genome_configs import _genome_config
    from scripts.gen_release_configs import _validate
    from nanoprot.data.builder import build_tokenizer
    cfg = _validate(_genome_config("gpt2", "S", 0, 4))
    assert cfg.tokenizer.name == "genome"
    assert cfg.data.dataset == "hg38"
    assert cfg.model.vocab_size == 10
    assert cfg.model.max_seq_len == 1024
    tok = build_tokenizer(cfg)
    assert type(tok).__name__ == "GenomeTokenizer"
    assert tok.get_vocab_size() == 10


def test_esm2_unaffected():
    """The additive harness edits must not change the existing esm2 tokenizer."""
    from nanoprot.tokenizers.esm2 import get_esm2_tokenizer
    assert get_esm2_tokenizer().get_vocab_size() == 33
