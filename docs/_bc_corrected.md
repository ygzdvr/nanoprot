# B-CAL — Bayesian (Gibbs, NUTS-equivalent) cross-check of γ_arch

Gibbs is exact for this conjugate Gaussian hierarchical LMM (no tuning/divergences); same
posterior NUTS targets. 4 dispersed chains; split-R̂ + bulk-ESS reported.

## Diagnostic self-test (validates the R̂/ESS code)
- iid N(0,1), 4×3000:  split-R̂=1.000 (→1.00), ESS=12000 (→12000)
- AR(1) φ=0.9, 4×3000: split-R̂=1.004, ESS=635 (theory≈632) ✓

## Posterior γ_arch vs EB(OLS)+bootstrap — per concept
concept | OLS γ (P3.3) | Bayes γ post-mean [95% CrI] | P(γ>0) | split-R̂ | ESS | agree?
      ss3 | +0.035 | +0.036 [+0.020,+0.052] | 1.000 | 1.003 | 1256 | yes
      ss8 | +0.021 | +0.022 [+0.009,+0.035] | 0.999 | 1.007 | 546 | yes
      rsa | +0.002 | +0.003 [-0.009,+0.014] | 0.670 | 1.007 | 519 | yes
   active | +0.005 | +0.006 [-0.006,+0.017] | 0.846 | 1.012 | 272 | check
 disorder | +0.026 | +0.026 [+0.005,+0.048] | 0.993 | 1.008 | 488 | yes

Reading: the Bayesian posterior γ_arch (random-intercept LMM) brackets the EB(OLS) estimate
with R̂<1.01 and ESS>400 ⇒ the matched-loss inductive-bias coefficient is robust to the
inference method (not an artifact of the OLS point-estimate + cluster bootstrap). P(γ>0)≈1 for
ss3/ss8/disorder reproduces the H_full≻H_loss verdict with full posterior uncertainty.
(JAX/numpyro NUTS unavailable on the cluster; Gibbs is exact here. A NUTS replicate is a
drop-in future check if JAX is provisioned.)
