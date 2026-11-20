# GQmanifold

A [Manopt](https://www.manopt.org) manifold factory for Riemannian optimization over the **generalized quadratic matrix manifold**

$$\mathrm{Gq}(P, Q) = \lbrace X \in \mathbb{R}^{n \times p} : X^\top Q X = P \rbrace,$$

where $Q \in \mathbb{R}^{n \times n}$ and $P \in \mathbb{R}^{p \times p}$ are invertible and carry the same symmetry, either both symmetric or both skew-symmetric.

One factory covers the classical Stiefel-type manifolds:

| $(P, Q)$ | manifold |
| --- | --- |
| $P$ and $Q$ symmetric positive definite | generalized Stiefel manifold, and the standard Stiefel manifold $\mathrm{St}(p,n)$ when $P = I_p$ and $Q = I_n$ |
| $P$ and $Q$ skew-symmetric | symplectic Stiefel manifold, with $P = J_p$ and $Q = J_n$ |
| $P = J$ with $J^2 = I_p$, $Q = A$ symmetric indefinite | indefinite Stiefel manifold $\mathrm{iSt}(J, A) = \mathrm{Gq}(J, A)$ |

The geometry is built from the $(P,Q)$-adjoint $A^{(P,Q)} = P^{-1} A^\top Q$ under a canonical-like metric, with closed-form gradient and Hessian conversions, three retractions (Cayley, quasi-geodesic, polar) with explicit domains, their second-order corrections, and matching vector transports.

This repository holds the code that reproduces the numerical experiments of two companion papers.

- Zhen Peng and Bin Gao, *Generalized Quadratic Matrix Manifolds: Riemannian Geometry and Computations*, arXiv:2610.28159, 2026. Referred to below as the **geometry paper**.
- Zhen Peng and Bin Gao, *Preconditioning for Quadratic Matrix Optimization via Constraint Reshaping*, arXiv:2611.26078, 2026. Referred to below as the **preconditioning paper**.

Both papers are available on arXiv: [2610.28159](https://arxiv.org/abs/2610.28159) (geometry) and [2611.26078](https://arxiv.org/abs/2611.26078) (preconditioning).

## Requirements

- MATLAB R2025a for the preconditioning experiments (`example_symm.m`, `example_skew.m`). The design LP uses `linprog` with the `dual-simplex-highs` algorithm; the scripts use `exportgraphics` and, for the geometry figures, `tiledlayout`. The geometry checks `check_retr.m` and `example_trace.m` also run under R2024b.
- MATLAB Optimization Toolbox. `example_symm.m` and `example_skew.m` solve the constraint-spectrum design with `linprog` and the secondary minimum-norm problem with `quadprog`.
- MATLAB Control System Toolbox. Both manifold factories use `lyap` to solve continuous-time Sylvester and Lyapunov equations.
- [Manopt](https://www.manopt.org) 8.0 on the path. The scripts call `trustregions`, `conjugategradient`, `stiefelfactory`, `hessianspectrum`, `checkgradient` and `checkhessian`; `check_retr.m` also relies on Manopt internals (`StoreDB`, `getCost`, `identify_linear_piece`), so it is tested against 8.0.

The toolboxes and Manopt are not bundled here; install them and add Manopt to the MATLAB path before running anything. CVX is not required.

## Quick start

Build the factory and minimize a quadratic over the constraint set.

```matlab
% Minimize trace(X'*A*X) over { X : X'*Q*X = P }.
% Q : n x n invertible, symmetric or skew-symmetric
% P : p x p invertible, with the SAME (skew-)symmetry as Q
n = 100; p = 10;
Q = gallery('lehmer', n);   % n x n, symmetric positive definite
P = eye(p);                 % p x p, symmetric positive definite
A = gallery('lehmer', n);   % symmetric cost matrix
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
| `native_design_fast.m` | Solves the exact affine design LP with `linprog` and the secondary minimum-norm QP with `quadprog`. Called by `example_symm.m` and `example_skew.m`. |
| `native_design_diagnostics.m` | Recomputes the LP and QP primal violations and KKT residuals after the setup timers stop. Called by the two preconditioning drivers. |
| `record_solver_termination.m` | Applies Manopt's stopping rules to the returned endpoint and records the stopping reason. Called by the two preconditioning drivers. |
| `flush_transp_guard_log.m` | Writes the deferred transport-guard counter CSV after the algorithm timers stop. |
| `check_retr.m` | Taylor-remainder check of the closed-form Riemannian gradient and Hessian, over the three retractions and the three configurations of $(P,Q)$, together with the second-order corrections. Reproduces the figure of Section 6.1 of the geometry paper (arXiv:2610.28159). Seed `rng(1)`, dimensions $n = 100$, $p = 10$. Writes `check_retr_n_100_p_10.pdf` and two CSV tables. |
| `example_trace.m` | Trace minimization $\min \lbrace \mathrm{tr}(X^\top A X) : X^\top Q X = P \rbrace$ in the symmetric, skew-symmetric and indefinite configurations, with both the conjugate gradient and the trust-region solver and every admissible retraction. Reproduces the figure and the tables of Section 6.2 of the geometry paper. Seed `rng(1)`, dimensions $n = 100$, $p = 10$. Writes `trace_RCG_n_100_p_10.pdf`, `trace_RTR_n_100_p_10.pdf`, one CSV table and one fixture MAT file per configuration. |
| `example_symm.m` | Preconditioning experiment, symmetric case. Compares the baseline $\mathrm{St}(p,n)$ with the reshaped constraint $\mathrm{Gq}(P,Q)$ under balanced and optimally designed constraint spectra, and with the Lagrangian-induced and left-right preconditioned metrics. Reproduces the symmetric half of the numerical experiments of the preconditioning paper. Seed `rng(0)`, dimensions $n = 100$, $p = 20$. Solves the design LP with `linprog` and the secondary QP with `quadprog`; CVX is not used. Writes `symm_*_n_100_p_20.pdf`, the matching CSV table, a timing CSV, a validation MAT file, and `legend_symm.pdf`. |
| `example_skew.m` | Preconditioning experiment, skew-symmetric case. Compares the baseline $\mathrm{Sp}(p,n)$ with the reshaped constraint $\mathrm{Gq}(P,Q)$ under balanced and optimally designed constraint spectra. Reproduces the skew-symmetric half of the numerical experiments of the preconditioning paper. Seed `rng(0)`, dimensions $n = 100$, $p = 20$. Solves the design LP with `linprog` and the secondary QP with `quadprog`; CVX is not used. Writes `skew_*_n_100_p_20.pdf`, the matching CSV table, a timing CSV, a validation MAT file, and `legend_skew.pdf`. |
| `results/` | Output directory. Every figure, table, timing record and log produced by the scripts lands here. It is created on demand and is not tracked. |

Each seed is set once at the top of its script, so a run reproduces the published numbers on a single random stream. In `example_symm.m` and `example_skew.m` the four objectives are named `eigenvalue`, `brockett`, `tridiagonal` and `quadratic`, and those names appear in the output file names. The native design stages are asserted not to advance the random stream.

The `*_timing.csv` files record one run's wall-clock breakdown. Times depend on the machine, the MATLAB release and the Optimization Toolbox solver, while the optimization trajectories and tables are reproducible up to floating-point and solver-version differences.

## Citation

If you use this code, please cite the geometry paper.

```bibtex
@article{peng2026generalized,
  title   = {Generalized Quadratic Matrix Manifolds: Riemannian Geometry and Computations},
  author  = {Peng, Zhen and Gao, Bin},
  journal = {arXiv preprint arXiv:2610.28159},
  year    = {2026},
  doi     = {10.48550/arXiv.2610.28159}
}
```

If you use the preconditioning experiments in `example_symm.m` or `example_skew.m`, please also cite the preconditioning paper.

```bibtex
@article{peng2026preconditioning,
  title   = {Preconditioning for Quadratic Matrix Optimization via Constraint Reshaping},
  author  = {Peng, Zhen and Gao, Bin},
  journal = {arXiv preprint arXiv:2611.26078},
  year    = {2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
