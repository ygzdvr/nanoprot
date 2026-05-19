"""
Protein-specific evaluation metrics for generated sequences.

Primary metric: amino acid frequency mean absolute deviation (AA MAD) from
natural UniRef50 frequencies, following ProtGPT2 (Ferruz et al., 2022).
"""
from collections import Counter

# Natural amino acid frequencies from UniRef50 (ProtGPT2 paper, Table 1)
NATURAL_AA_FREQ = {
    'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0545, 'C': 0.0137,
    'E': 0.0675, 'Q': 0.0393, 'G': 0.0707, 'H': 0.0227, 'I': 0.0596,
    'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386, 'P': 0.0470,
    'S': 0.0657, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0687,
}
STANDARD_AAS = sorted(NATURAL_AA_FREQ.keys())

# Full IUPAC alphabet (20 standard + 5 ambiguity codes)
VALID_AA = set("ARNDCEQGHILKMFPSTWYVXBUOZ")


def compute_aa_frequencies(sequences):
    """Compute amino acid frequencies from a list of protein sequences."""
    total = Counter()
    for seq in sequences:
        total.update(seq)
    n = sum(total[aa] for aa in STANDARD_AAS)
    if n == 0:
        return {aa: 0.0 for aa in STANDARD_AAS}
    return {aa: total[aa] / n for aa in STANDARD_AAS}


def aa_frequency_deviation(generated_sequences):
    """Compute mean absolute deviation of AA frequencies from natural UniRef50."""
    gen_freq = compute_aa_frequencies(generated_sequences)
    deviations = [abs(gen_freq[aa] - NATURAL_AA_FREQ[aa]) for aa in STANDARD_AAS]
    return sum(deviations) / len(deviations), gen_freq


def evaluate_generated_proteins(sequences):
    """Full evaluation suite for generated protein sequences.

    Returns dict with:
        aa_mad: mean absolute deviation of AA frequencies from natural
        aa_freq: dict of per-AA frequencies in generated sequences
        mean_length: mean sequence length
        std_length: std of sequence lengths
        fraction_valid: fraction of sequences using only standard 20 AAs
    """
    results = {}
    # 1. Amino acid frequencies
    mad, gen_freq = aa_frequency_deviation(sequences)
    results['aa_mad'] = mad
    results['aa_freq'] = gen_freq
    # 2. Length statistics
    lengths = [len(s) for s in sequences]
    results['mean_length'] = sum(lengths) / len(lengths)
    results['std_length'] = (sum((l - results['mean_length'])**2 for l in lengths) / len(lengths)) ** 0.5
    # 3. Fraction of valid sequences (only standard 20 AAs)
    valid = sum(1 for s in sequences if all(c in STANDARD_AAS for c in s))
    results['fraction_valid'] = valid / len(sequences)
    return results
