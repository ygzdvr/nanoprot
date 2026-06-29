# B-CAL — Bayesian (Gibbs, NUTS-equivalent) cross-check of γ_arch

Gibbs is exact for this conjugate Gaussian hierarchical LMM (no tuning/divergences); same
posterior NUTS targets. 4 dispersed chains; split-R̂ + bulk-ESS reported.

## Diagnostic self-test (validates the R̂/ESS code)
- iid N(0,1), 4×3000:  split-R̂=1.000 (→1.00), ESS=12000 (→12000)
- AR(1) φ=0.9, 4×3000: split-R̂=1.004, ESS=635 (theory≈632) ✓

## Posterior γ_arch vs EB(OLS)+bootstrap — per concept
concept | OLS γ (P3.3) | Bayes γ post-mean [95% CrI] | P(γ>0) | split-R̂ | ESS | agree?
      ss3 | +0.035 | +0.035 [+0.019,+0.052] | 1.000 | 1.000 | 4041 | yes
      ss8 | +0.021 | +0.021 [+0.008,+0.035] | 0.999 | 1.000 | 1791 | yes
      rsa | +0.002 | +0.002 [-0.010,+0.014] | 0.634 | 1.000 | 1720 | yes
   active | +0.005 | +0.005 [-0.007,+0.017] | 0.809 | 1.001 | 1083 | yes
 disorder | +0.026 | +0.026 [+0.004,+0.047] | 0.989 | 1.000 | 1603 | yes

Reading: the Bayesian posterior γ_arch (random-intercept LMM) brackets the EB(OLS) estimate
with R̂<1.01 and ESS>400 ⇒ the matched-loss inductive-bias coefficient is robust to the
inference method (not an artifact of the OLS point-estimate + cluster bootstrap). P(γ>0)≈1 for
ss3/ss8/disorder reproduces the H_full≻H_loss verdict with full posterior uncertainty.
(JAX/numpyro NUTS unavailable on the cluster; Gibbs is exact here. A NUTS replicate is a
drop-in future check if JAX is provisioned.)
