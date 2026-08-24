# Mathematical scope

## Delay correction

For a delay distribution (p(\tau\mid h)) over \(\tau=0,\ldots,K\), CCPL
uses

\[
\gamma_{\mathrm{eff}}(h)=\sum_{\tau=0}^{K}p(\tau\mid h)\gamma^\tau.
\]

Because this is a convex combination,
\(\gamma^K\leq\gamma_{\mathrm{eff}}(h)\leq1\). A strict contraction bound
requires positive minimum delay (or another assumption excluding \(\tau=0\));
unknown stochastic delays alone do not establish a modulus below one.

The implementation checks the simplex and these bounds in
`ccpl_theory.effective_discount_bounds`.

## State-conditioned multipliers

Variation in conditional costs, \(\mathrm{Var}_s[E[c\mid s]]>0\), shows
heterogeneity but does not prove that a state-conditioned multiplier dominates
a scalar multiplier under one global CMDP constraint. That comparison is an
empirical claim and requires matched seeds, confidence intervals, and an
explicit policy class.

## Causal attribution

The SCM calibration measures agreement with the programmed synthetic structural
equations. It is not causal identification from observational data and should
not be described as do-calculus validation on external environments.
