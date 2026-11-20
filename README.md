# GQmanifold

A [Manopt](https://www.manopt.org) manifold factory for Riemannian optimization over the **generalized quadratic matrix manifold**

$$\mathrm{Gq}(P, Q) = \lbrace X \in \mathbb{R}^{n \times p} : X^\top Q X = P \rbrace,$$

where $Q \in \mathbb{R}^{n \times n}$ and $P \in \mathbb{R}^{p \times p}$ are invertible and carry the same symmetry, either both symmetric or both skew-symmetric. Here $p \le n$; in the skew-symmetric case both dimensions are even.

One factory covers the classical Stiefel-type manifolds and the quadratic matrix Lie groups:

| $(P, Q)$ | manifold |
| --- | --- |
| $P$ and $Q$ symmetric positive definite | generalized Stiefel manifold, and the standard Stiefel manifold $\mathrm{St}(p,n)$ when $P = I_p$ and $Q = I_n$ |
| $P$ and $Q$ skew-symmetric | symplectic Stiefel manifold, with $P = J_p$ and $Q = J_n$ |
| $P = J$ with $J^2 = I_p$, $Q = A$ symmetric indefinite | indefinite Stiefel manifold $\mathrm{iSt}(J, A) = \mathrm{Gq}(J, A)$ |
| $p = n$, $P = Q$ | quadratic matrix Lie group $\mathrm{QLie}(Q) = \mathrm{Gq}(Q, Q)$ |

Under a canonical-like metric the factory provides closed-form gradient and Hessian conversions, the Cayley, quasi-geodesic and polar retractions with their second-order corrections, and matching vector transports.

## Companion papers

The code reproduces the numerical experiments in:

- Zhen Peng and Bin Gao, *Generalized Quadratic Matrix Manifolds: Riemannian Geometry and Computations*, [arXiv:2610.28159](https://arxiv.org/abs/2610.28159), 2026.
- Zhen Peng and Bin Gao, *Preconditioning for Quadratic Matrix Optimization via Constraint Reshaping*, [arXiv:2611.26078](https://arxiv.org/abs/2611.26078), 2026.

## Requirements

- MATLAB R2025a for the preconditioning experiments (`example_symm.m`, `example_skew.m`); `check_retr.m` and `example_trace.m` also run under R2024b.
- MATLAB Optimization Toolbox (`linprog`, `quadprog`) and Control System Toolbox (`lyap`).
- [Manopt](https://www.manopt.org) 8.0 on the path.

CVX is not required.

## Quick start

```matlab
% Minimize trace(X'*A*X) over { X : X'*Q*X = P }.
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

checkgradient(problem);   % first-order Taylor test
checkhessian(problem);    % second-order Taylor test, expected slope 3
```

## Running the experiments

Start MATLAB in this directory. Run `check_retr` or `example_trace` for the geometry paper, and `example_symm` or `example_skew` for the preconditioning paper. Each script sets its random seed at the top and writes its outputs to `results/`, which is created on demand. Timings depend on the machine, and iteration counts can vary slightly between MATLAB releases.

## Citation

If you use the geometry code, please cite

```bibtex
@article{peng2026generalized,
  title   = {Generalized Quadratic Matrix Manifolds: Riemannian Geometry and Computations},
  author  = {Peng, Zhen and Gao, Bin},
  journal = {arXiv preprint arXiv:2610.28159},
  year    = {2026}
}
```

If you use the preconditioning experiments, please cite

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
