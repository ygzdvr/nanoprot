# Trajectory emergence report (Pillar 2)

Δ = learned − fixed-baseline; t_* in training steps (seed-mean). AUEC = ∫Δ d·log(step). gpt2−mamba<0 in t_onset ⇒ gpt2 emerges earlier.

## active / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 355 | 326 | 619 | +0.045 | +0.045 |
| gpt2 | M | 408 | 408 | 1664 | +0.058 | +0.110 |
| mamba | S | 250 | 250 | 443 | +0.041 | +0.039 |
| mamba | M | 746 | 657 | 1568 | +0.035 | +0.029 |
|  ↳ gpt2−mamba onset @ S | | +105 steps (mamba earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -339 steps (gpt2 earlier) | | | | |

## disorder / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 1 | 13 | 337 | +0.216 | +0.860 |
| gpt2 | M | 2 | 10 | 167 | +0.154 | +0.803 |
| mamba | S | 2 | 62 | 151 | +0.143 | +0.457 |
| mamba | M | 8 | 56 | 2845 | +0.164 | +0.613 |
|  ↳ gpt2−mamba onset @ S | | -1 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -7 steps (gpt2 earlier) | | | | |

## rsa / netsurfp  (metric=r2)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 7 | 229 | 598 | +0.059 | +0.125 |
| gpt2 | M | 8 | 456 | 2768 | +0.092 | +0.219 |
| mamba | S | 14 | 122 | 471 | +0.051 | +0.127 |
| mamba | M | 48 | 228 | 3095 | +0.070 | +0.193 |
|  ↳ gpt2−mamba onset @ S | | -7 steps (gpt2 earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -40 steps (gpt2 earlier) | | | | |

## signal_peptide / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 4 | 7 | 13 | +0.496 | +2.381 |
| gpt2 | M | 4 | 6 | 20 | +0.500 | +3.138 |
| gpt2 | L | 3 | 6 | 18 | +0.492 | +3.450 |
| esm2 | M | 1 | 103 | 201 | +0.370 | +1.421 |

## ss3 / netsurfp  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 2 | 8 | 230 | +0.164 | +0.558 |
| gpt2 | M | 2 | 95 | 1886 | +0.230 | +0.866 |
| mamba | S | 1 | 22 | 253 | +0.114 | +0.389 |
| mamba | M | 1 | 76 | 407 | +0.156 | +0.676 |
|  ↳ gpt2−mamba onset @ S | | +1 steps (mamba earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | +1 steps (mamba earlier) | | | | |

## ss8 / netsurfp  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 2 | 18 | 409 | +0.088 | +0.259 |
| gpt2 | M | 2 | 86 | 1326 | +0.118 | +0.471 |
| mamba | S | 1 | 2 | 142 | +0.067 | +0.251 |
| mamba | M | 4 | 204 | 3473 | +0.082 | +0.252 |
|  ↳ gpt2−mamba onset @ S | | +1 steps (mamba earlier) | | | | |
|  ↳ gpt2−mamba onset @ M | | -1 steps (gpt2 earlier) | | | | |

## transmembrane / swissprot  (metric=macro_f1)

| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |
|------|-------|---------|-----|-----|---------|------|
| gpt2 | S | 2 | 10 | 106 | +0.289 | +1.182 |
| gpt2 | M | 1 | 6 | 174 | +0.260 | +1.367 |
| gpt2 | L | 2 | 7 | 109 | +0.240 | +1.574 |
| esm2 | M | 1 | 7 | 120 | +0.184 | +1.067 |

