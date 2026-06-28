# P3.3 — capability-scaling: emergence-time N-scaling + inductive-bias coefficient

Δ = learned − random-init (learned structure); t50 = log-compute at ½·Δ_final (observed,
no saturation assumption); LOGN(S,M,L)=(17.4,18.8,20.7). Seed-bootstrap CIs.

## 1–2. Emergence time t50 and asymptote Δ_final by scale; slope vs logN
concept · arch | t50 @S/M/L | dt50/dlogN | Δ_final @S/M/L | dΔ/dlogN
      ss3 · gpt2  | 20.8/24.1/26.1 | +1.55 | 0.167/0.227/0.280 | +0.034
      ss3 · mamba | 23.5/23.8/28.7 | +1.63 | 0.125/0.149/0.212 | +0.027
      ss8 · gpt2  | 22.1/24.5/27.2 | +1.53 | 0.081/0.106/0.186 | +0.032
      ss8 · mamba | 23.2/25.7/28.6 | +1.64 | 0.070/0.070/0.108 | +0.012
      rsa · gpt2  | 24.6/26.3/29.2 | +1.40 | 0.058/0.086/0.150 | +0.028
      rsa · mamba | 25.3/27.0/29.0 | +1.11 | 0.050/0.067/0.119 | +0.021
   active · gpt2  | 25.0/26.5/27.3 | +0.69 | 0.040/0.061/0.074 | +0.010
   active · mamba | 26.0/27.5/26.8 | +0.21 | 0.034/0.029/0.055 | +0.007
 disorder · gpt2  | 21.1/23.3/25.6 | +1.35 | 0.205/0.160/0.137 | -0.020
 disorder · mamba | 23.8/25.6/26.7 | +0.85 | 0.138/0.150/0.130 | -0.003

## 3. Inductive bias: Δ ~ β0 + β_L·(−bpr) + scale-FE + γ_arch·[gpt2]  (AR pair, all ckpts)
γ_arch>0 ⇒ gpt2 has MORE learned structure than mamba at matched measured-loss & scale.
concept | γ_arch [95% CI] | β_L | ΔAIC(H_loss−H_full) | n | H_full wins?
      ss3 | +0.035 [+0.029,+0.043] | +0.136 | +47.2 | 372 | yes
      ss8 | +0.021 [+0.015,+0.028] | +0.071 | +56.6 | 372 | yes
      rsa | +0.002 [-0.002,+0.005] | +0.064 | -1.2 | 372 | no
   active | +0.005 [+0.001,+0.009] | +0.043 | +5.6 | 372 | yes
 disorder | +0.026 [+0.003,+0.049] | +0.104 | +29.4 | 372 | yes

Driving by MEASURED loss makes γ_arch robust to the mamba replicated-slice offset. NUTS
cross-check + the L(N,D) α/β surface (non-identifiable from the noisy sweep) remain TODO.
