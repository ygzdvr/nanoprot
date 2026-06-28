# Trajectory emergence report (Pillar 2)

Δ = learned − fixed-baseline; t_* in training steps (seed-mean). AUEC = ∫Δ d·log(step). gpt2−mamba<0 in t_onset ⇒ gpt2 emerges earlier.

## active / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 355 | 326 | 619 | +0.045 | +0.045 |
| gpt2 | M | 408 | 408 | 1664 | +0.058 | +0.110 |
| gpt2 | L | 175 | 203 | 364 | +0.064 | +0.305 |
| mamba | S | 250 | 250 | 443 | +0.041 | +0.039 |
| mamba | M | 746 | 657 | 1568 | +0.035 | +0.029 |
| mamba | L | 57 | 76 | 861 | +0.054 | +0.244 |
| esm2 | M | 383 | 488 | 1718 | +0.087 | +0.165 |
|  ↳ gpt2−mamba onset @ S | | +105 steps (mamba earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -339 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ L | | +118 steps (mamba earlier) | | | | |

## disorder / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 1 | 13 | 337 | +0.216 | +0.860 |
| gpt2 | M | 2 | 10 | 167 | +0.154 | +0.803 |
| gpt2 | L | 6 | 56 | 146 | +0.129 | +0.840 |
| mamba | S | 2 | 62 | 151 | +0.143 | +0.457 |
| mamba | M | 8 | 56 | 2845 | +0.164 | +0.613 |
| mamba | L | 3 | 44 | 151 | +0.124 | +0.840 |
| esm2 | M | 35 | 88 | 1635 | +0.129 | +0.460 |
|  ↳ gpt2−mamba onset @ S | | -1 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -7 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ L | | +3 steps (mamba earlier) | | | | |

## rsa / netsurfp  (metric=r2)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 7 | 229 | 598 | +0.059 | +0.125 |
| gpt2 | M | 8 | 456 | 2768 | +0.092 | +0.219 |
| gpt2 | L | 9 | 2233 | 13718 | +0.169 | +0.405 |
| mamba | S | 14 | 122 | 471 | +0.051 | +0.127 |
| mamba | M | 48 | 228 | 3095 | +0.070 | +0.193 |
| mamba | L | 3 | 1112 | 14432 | +0.132 | +0.349 |
| esm2 | M | 73 | 436 | 2654 | +0.172 | +0.344 |
|  ↳ gpt2−mamba onset @ S | | -7 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -40 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ L | | +5 steps (mamba earlier) | | | | |

## signal_peptide / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 4 | 7 | 13 | +0.496 | +2.381 |
| gpt2 | M | 4 | 6 | 20 | +0.500 | +3.138 |
| gpt2 | L | 3 | 6 | 18 | +0.475 | +3.790 |
| mamba | L | 2 | 61 | 72 | +0.329 | +2.011 |
| esm2 | M | 1 | 153 | 225 | +0.352 | +1.218 |
|  ↳ gpt2−mamba onset @ L | | +1 steps (mamba earlier) | | | | |

## ss3 / netsurfp  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 2 | 8 | 230 | +0.164 | +0.558 |
| gpt2 | M | 2 | 95 | 1886 | +0.230 | +0.866 |
| gpt2 | L | 4 | 147 | 9618 | +0.310 | +1.325 |
| mamba | S | 1 | 22 | 253 | +0.114 | +0.389 |
| mamba | M | 1 | 76 | 407 | +0.156 | +0.676 |
| mamba | L | 2 | 740 | 14432 | +0.243 | +0.778 |
| esm2 | M | 2 | 348 | 1243 | +0.261 | +0.770 |
|  ↳ gpt2−mamba onset @ S | | +1 steps (mamba earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | +1 steps (mamba earlier) | | | | |
|  ↳ gpt2−mamba onset @ L | | +2 steps (mamba earlier) | | | | |

## ss8 / netsurfp  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 2 | 18 | 409 | +0.088 | +0.259 |
| gpt2 | M | 2 | 86 | 1326 | +0.118 | +0.471 |
| gpt2 | L | 3 | 499 | 12047 | +0.198 | +0.732 |
| mamba | S | 1 | 2 | 142 | +0.067 | +0.251 |
| mamba | M | 4 | 204 | 3473 | +0.082 | +0.252 |
| mamba | L | 74 | 812 | 11546 | +0.115 | +0.380 |
| esm2 | M | 26 | 383 | 2292 | +0.151 | +0.413 |
|  ↳ gpt2−mamba onset @ S | | +1 steps (mamba earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -1 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ L | | -71 steps (gpt2 earlier) | | | | |

## transmembrane / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 2 | 10 | 106 | +0.289 | +1.182 |
| gpt2 | M | 1 | 6 | 174 | +0.260 | +1.367 |
| gpt2 | L | 3 | 38 | 317 | +0.313 | +1.772 |
| mamba | L | 2 | 29 | 4922 | +0.176 | +1.065 |
| esm2 | M | 4 | 89 | 180 | +0.268 | +1.025 |
|  ↳ gpt2−mamba onset @ L | | +1 steps (mamba earlier) | | | | |

