# GQmanifold

A [Manopt](https://www.manopt.org) manifold factory for Riemannian optimization over the **generalized quadratic matrix manifold**

$$\mathrm{Gq}(P, Q) = \lbrace X \in \mathbb{R}^{n \times p} : X^\top Q X = P \rbrace,$$

where $Q \in \mathbb{R}^{n \times n}$ and $P \in \mathbb{R}^{p \times p}$ are invertible and carry the same symmetry, either both symmetric or both skew-symmetric.

One factory covers the classical Stiefel-type manifolds:

| $(Q, P)$ | manifold |
| --- | --- |
| symmetric positive definite | generalized Stiefel manifold, and the standard Stiefel manifold $\mathrm{St}(p,n)$ when $Q = I_n$ and $P = I_p$ |
| skew-symmetric | symplectic Stiefel manifold, with $Q = J_n$ and $P = J_p$ |
| $Q$ symmetric indefinite, $P = \mathrm{diag}(\pm 1)$ with $P^2 = I$ | indefinite Stiefel manifold $\mathrm{iSt}(A, J) = \mathrm{Gq}(J, A)$ |

The geometry is built from the $(P,Q)$-adjoint $A^{(P,Q)} = P^{-1} A^\top Q$ under a canonical-like metric, with closed-form gradient and Hessian conversions, three feasible retractions (Cayley, quasi-geodesic, polar), their second-order corrections, and matching vector transports.

This repository holds the code that reproduces the numerical sections of two companion papers.

- Zhen Peng and Bin Gao, *Generalized Quadratic Matrix Manifolds: Geometry and Computations*, preprint, 2026. Referred to below as the **geometry paper**.
- Zhen Peng and Bin Gao, *Preconditioning for Ill-Conditioned Quadratic Optimization via Reshaping Constraints*, preprint, 2026. Referred to below as the **preconditioning paper**.

Both papers are preprints at the time of this release. The arXiv identifiers will be added to the BibTeX entries below once they are assigned.

## Requirements

- MATLAB R2020a or later. The example scripts use `tiledlayout` and `exportgraphics`.
- MATLAB Control System Toolbox. The manifold factories use `lyap` to solve continuous-time Sylvester and Lyapunov equations.
- [Manopt](https://www.manopt.org) 8.0 or later on the path. The scripts call `trustregions`, `conjugategradient`, `stiefelfactory`, `hessianspectrum`, `checkgradient` and `checkhessian`.
- [CVX](https://cvxr.com) on the path, needed only by `example_symm.m` and `example_skew.m`, which solve a linear program to design the constraint spectra.

The Control System Toolbox must be installed with MATLAB. Manopt and CVX are not bundled here; install them and add them to the MATLAB path before running anything.

## Quick start

Build the factory and minimize a quadratic over the constraint set.

```matlab
% Minimize trace(X'*A*X) over { X : X'*Q*X = P }.
% Q : n x n invertible, symmetric or skew-symmetric
% P : p x p invertible, with the SAME (skew-)symmetry as Q
M = generalfactory(Q, P);          % canonical-like metric, beta = 1
% M = generalfactory(Q, P, beta);  % custom metric scaling beta > 0

problem.M     = M;
problem.cost  = @(X) trace(X'*A*X);
problem.egrad = @(X) 2*A*X;          % Euclidean gradient
problem.ehess = @(X, Xdot) 2*A*Xdot; % Euclidean Hessian

X0 = M.rand();                       % random feasible starting point
[X, cost, info] = trustregions(problem, X0); % or conjugategradient
```

The Euclidean-to-Riemannian conversions are built in, so the closed-form gradient and Hessian can be checked with Manopt's own tools.

```matlab
checkgradient(problem);   % first-order Taylor test
checkhessian(problem);    % second-order Taylor test, expected slope 3
```

To reproduce a published experiment, start MATLAB in this directory and run one of the scripts by name, for example `example_trace`. Every script writes into `results/`, which it creates if it does not exist.

## Files

| File | Purpose |
| --- | --- |
| `generalfactory.m` | The manifold factory for $\mathrm{Gq}(P,Q)$. Supplies the canonical-like metric, the tangent-space projection, the closed-form gradient and Hessian conversions, the Cayley, quasi-geodesic and polar retractions with their second-order corrections and vector transports, the exponential map, and random point and tangent vector generators. Every script below builds its manifolds through this file. |
| `symplecticfactory.m` | Standalone factory for the symplectic Stiefel manifold $\mathrm{Sp}(2p,2n)$, with a selectable Euclidean or canonical-like metric and a closed-form Riemannian Hessian. Serves as the baseline in `example_skew.m`. |
| `YoulaDecomposition.m` | Real Youla decomposition of a skew-symmetric matrix. Called by the skew-symmetric branch of `M.rand` in `generalfactory.m` and by `example_skew.m`. |
| `check_retr.m` | Taylor-remainder check of the closed-form Riemannian gradient and Hessian, over the three retractions and the three configurations of $(P,Q)$, together with the second-order corrections. Reproduces the figure of Section 5.1 of the geometry paper. Seed `rng(1)`, dimensions $n = 100$, $p = 10$. Writes `check_retr_n_100_p_10.pdf` and two CSV tables. |
| `example_trace.m` | Trace minimization $\min \lbrace \mathrm{tr}(X^\top A X) : X^\top Q X = P \rbrace$ in the symmetric, skew-symmetric and indefinite configurations, with both the conjugate gradient and the trust-region solver and every admissible retraction. Reproduces the figure and the tables of Section 5.2 of the geometry paper. Seed `rng(1)`, dimensions $n = 100$, $p = 10$. Writes `trace_RCG_n_100_p_10.pdf`, `trace_RTR_n_100_p_10.pdf` and one CSV table per configuration. |
| `example_symm.m` | Preconditioning experiment, symmetric case. Compares the baseline $\mathrm{St}(p,n)$ with the reshaped constraint $\mathrm{Gq}(P,Q)$ under balanced and optimally designed constraint spectra, and with the Lagrangian-induced and left-right preconditioned metrics. Reproduces the symmetric half of Section 6.1 of the preconditioning paper. Seed `rng(0)`, dimensions $n = 100$, $p = 20$. Writes `symm_*_n_100_p_20.pdf` and the matching CSV tables. Requires CVX. |
| `example_skew.m` | Preconditioning experiment, skew-symmetric case. Compares the baseline $\mathrm{Sp}(p,n)$ with the reshaped constraint $\mathrm{Gq}(P,Q)$ under balanced and optimally designed constraint spectra. Reproduces the skew-symmetric half of Section 6.1 of the preconditioning paper. Seed `rng(0)`, dimensions $n = 100$, $p = 20$. Writes `skew_*_n_100_p_20.pdf` and the matching CSV tables. Requires CVX. |
| `results/` | Output directory. Every figure, table and log produced by the scripts lands here. |

Each seed is set once at the top of its script, so a run reproduces the published numbers on a single random stream. In `example_symm.m` and `example_skew.m` the four objectives are named `eigenvalue`, `brockett`, `tridiagonal` and `quadratic`, and those names appear in the output file names.

## Citation

If you use this code, please cite the geometry paper.

```bibtex
@article{peng2026generalized,
  title   = {Generalized Quadratic Matrix Manifolds: Geometry and Computations},
  author  = {Peng, Zhen and Gao, Bin},
  journal = {arXiv preprint},
  year    = {2026}
}
```

If you use the preconditioning experiments in `example_symm.m` or `example_skew.m`, please also cite the preconditioning paper.

```bibtex
@article{peng2026preconditioning,
  title   = {Preconditioning for Ill-Conditioned Quadratic Optimization via Reshaping Constraints},
  author  = {Peng, Zhen and Gao, Bin},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
