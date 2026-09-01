function M = symplecticfactory(n, p, metric, beta)
% SYMPLECTICFACTORY  Symplectic Stiefel manifold Sp(2p,2n) for Manopt, with a
%                    selectable Riemannian metric (Euclidean or canonical-like).
%
% function M = symplecticfactory(n, p)                 % Euclidean (default)
% function M = symplecticfactory(n, p, 'euclidean')
% function M = symplecticfactory(n, p, 'canonical')    % canonical-like, beta=1/2
% function M = symplecticfactory(n, p, 'canonical', beta)
%
% Manifold:  Sp(2p,2n) = { X in R^{2n x 2p} : X' J_{2n} X = J_{2p} },
%   J_{2m} = [0 I_m; -I_m 0],   dim = 4np - p(2p-1).
%
% Two metrics from the tractable family of
%   Gao, Son, Stykel, "Symplectic Stiefel manifold: tractable metrics,
%   second-order geometry and Newton's methods" (arXiv:2406.14299):
%
%   'euclidean'  : g_e(Z1,Z2) = <Z1,Z2>            (M_X = I_{2n})
%        - projection / gradient : eqs. (23)-(25) with M = I
%        - Riemannian Hessian    : Corollary 1
%
%   'canonical'  : g_c(Z1,Z2) = <Z1, M_{X,c,beta} Z2>,
%        M_{X,c,beta} = (1/beta) J_{2n}X X' J_{2n}' + (I - X(X'X)^{-1}X'),
%        M_{X,c,beta}^{-1} = beta X X' + P_X P_X',  P_X = I - X J_{2p} X' J_{2n}'.
%        - projection            : eq. (19)
%        - gradient              : eq. (20)
%        - Riemannian Hessian    : Theorem 2
%
% Both Hessians were verified against a finite-difference Riemannian Hessian
% at a critical point (rel. err ~1e-10) and shown self-adjoint in their metric
% (asym ~1e-12 to 1e-14).  The tangent space and the Cayley retraction are
% metric-independent and shared by both branches.

    skew = @(A) 0.5*(A - A');
    sym  = @(A) 0.5*(A + A');

    if nargin < 2 || isempty(p),      p = n;             end
    if nargin < 3 || isempty(metric), metric = 'euclidean'; end
    if nargin < 4 || isempty(beta),    beta = 0.5;          end
    metric = lower(metric);
    is_canon = strncmp(metric, 'c', 1);   % 'canonical' / 'canonical-like'
    assert(n >= p, 'The dimension n must be larger than the dimension p.');

    I2n = eye(2*n);
    J2p = [zeros(p,p) eye(p); -eye(p) zeros(p,p)];

    if is_canon
        M.name = @() sprintf('Sp(%d,%d), canonical-like metric (beta=%.3g)', n, p, beta);
    else
        M.name = @() sprintf('Sp(%d,%d), Euclidean metric', n, p);
    end
    M.dim = @() 4*n*p - p*(2*p-1);

    % Block actions:  J_{2n}X,  J_{2n}'Y,  X J_{2p}.
    Jn = @(X) [X(n+1:end, :); -X(1:n, :)];     % J_{2n} X
    Jt = @(Y) [-Y(n+1:end, :); Y(1:n, :)];     % J_{2n}' Y
    XJ = @(X) [-X(:, p+1:end), X(:, 1:p)];     % X J_{2p}
    PXobl = @(X) I2n - XJ(X) * (Jn(X))';       % P_X = I - X J_{2p} X' J_{2n}'

    % ==================================================================
    % METRIC, PROJECTION, GRADIENT, HESSIAN  (branch on the chosen metric)
    % ==================================================================
    if is_canon
        % ---------------- canonical-like metric ----------------
        M.inner = @(X, Z1, Z2) Z1(:)' * reshape(applyMc(X, Z2), [], 1);
        M.proj  = @proj_c;
        M.egrad2rgrad = @egrad2rgrad_c;
        M.ehess2rhess = @ehess2rhess_c;
    else
        % ---------------- Euclidean metric ----------------
        M.inner = @(X, Z1, Z2) Z1(:)' * Z2(:);
        M.proj  = @proj_e;
        M.egrad2rgrad = @egrad2rgrad_e;
        M.ehess2rhess = @ehess2rhess_e;
    end
    M.norm    = @(X, Z) sqrt(max(0, M.inner(X, Z, Z)));
    M.tangent = M.proj;
    M.dist    = @(x, y) error('symplecticfactory.dist not implemented.');

    % ---- canonical metric operator M_{X,c,beta} (for inner) ----
    function MZ = applyMc(X, Z)
        JX = Jn(X);
        Piperp = Z - X*((X'*X) \ (X'*Z));             % (I - X(X'X)^{-1}X') Z
        MZ = (1/beta)*(JX * (JX'*Z)) + Piperp;         % since J X X' J' = (JX)(JX)'
    end
    function MiZ = applyMcinv(X, Z)
        PX = PXobl(X);
        MiZ = beta*(X*(X'*Z)) + PX*(PX'*Z);            % beta X X' Z + P_X P_X' Z
    end

    % ---- projections ----
    function Yp = proj_c(X, Y)                          % eq. (19)
        Yp = Y - XJ(X) * skew(X' * Jt(Y));
    end
    function Yp = proj_e(X, Y)                          % eq. (23) with M = I
        A  = X'*X;
        Om = lyap(A, -2*skew(X' * Jt(Y)));
        Yp = Y - Jn(X) * skew(Om);
    end

    % ---- gradients ----
    function rg = egrad2rgrad_c(X, G)                   % eq. (20)
        PX = PXobl(X);
        rg = beta * XJ(X) * sym(J2p' * X' * G) + PX*(PX'*G);
    end
    function rg = egrad2rgrad_e(X, G)                   % eq. (25) with M = I
        A  = X'*X;
        Om = skew(lyap(A, -2*skew(X' * Jt(G))));
        rg = G - Jn(X) * Om;
    end

    % ---- Riemannian Hessians ----
    function rhess = ehess2rhess_c(X, egrad, ehess, Z)  % Theorem 2
        PX = PXobl(X);
        XX = X'*X;
        g  = egrad;
        Hg = ehess;
        blockA = applyMcinv(X, Hg) ...
               - beta * Z * J2p * skew(J2p' * X' * g) ...
               + 2 * sym( beta*X*Z' - 2*skew(X*J2p*Z') * J2nt() * PX' ) * g ...
               + beta * X * sym( X'*(Jn(Z))*g'*X*J2p - Z'*g );
        t1 = PX' * Jn( Z * sym(g'*X*J2p) + (1/beta)*PX'*g * (Z'*(Jn(X))) );
        t2 = - Jn(X*J2p) * sym( Z'*PX'*g );
        t3 = - Z * skew( J2p*X'*(Jn(PX'*g)) + beta*J2p*sym(J2p'*X'*g) );
        t4 = - PX'*g * skew( XX \ (X'*Z) );
        rhess = proj_c(X, blockA) + PX*(t1 + t2 + t3 + t4);
    end
    function rhess = ehess2rhess_e(X, egrad, ehess, Z)  % Corollary 1
        A   = X'*X;
        Om  = skew(lyap(A, -2*skew(X' * Jt(egrad))));
        Th  = skew(lyap(A, -2*skew(X' * Jt(ehess) - X'*Z*Om)));
        rhess = ehess - Jn(Z)*Om - Jn(X)*Th;
    end

    % Matrix form of J_{2n}' used as a left factor in the Theorem-2 terms.
    function T = J2nt()
        T = [zeros(n,n) -eye(n); eye(n) zeros(n,n)];    % = J_{2n}'
    end

    % ==================================================================
    % SHARED: Cayley retraction, random point/vector, helpers
    % ==================================================================
    M.retr = @retraction_cl;
    M.retr_cl = @retraction_cl;
    function Y = retraction_cl(X, U, t)
        if nargin < 3, t = 1; end
        XJm = XJ(X);
        JX  = Jn(X);
        GX  = I2n - 0.5*(XJm*JX');
        SXZ = 2*sym(GX * U * XJm');
        SXZJ  = [-SXZ(:, n+1:end), SXZ(:, 1:n)];
        SXZJX = SXZ * JX;
        Y = (I2n - (0.5*t)*SXZJ) \ (X + (0.5*t)*SXZJX);
    end

    M.rand = @randmfd;
    function Y = randmfd()
        W = randn(2*p, 2*p); W = W'*W + 0.1*eye(2*p);
        E = expm([W(p+1:end, :); -W(1:p, :)]);
        Y = [E(1:p, :); zeros(n-p, 2*p); E(p+1:end, :); zeros(n-p, 2*p)];
    end

    M.randvec = @randomvec;
    function U = randomvec(X)
        U = M.proj(X, randn(2*n, 2*p));
        U = U / M.norm(X, U);
    end

    M.lincomb = @matrixlincomb;
    M.zerovec = @(x) zeros(2*n, 2*p);
    M.transp  = @(x1, x2, d) M.proj(x2, d);
    M.vec     = @(x, u_mat) u_mat(:);
    M.mat     = @(x, u_vec) reshape(u_vec, [2*n, 2*p]);
    M.vecmatareisometries = @() ~is_canon;   % isometry only for the Euclidean metric

    M.is_on_mfd = @(X) norm(X' * Jn(X) - J2p, 'fro') < 1e-10;
end