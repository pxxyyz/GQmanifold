function M = generalfactory(Q, P, beta)
% GENERALFACTORY  Manopt factory for the generalized quadratic matrix manifold
%                 Gq(P,Q) = { X in R^{n x p} : X'*Q*X = P }.
%
% function M = generalfactory(Q, P)         % canonical-like metric, beta = 1
% function M = generalfactory(Q, P, beta)   % metric scaling beta > 0
%
% Q (n x n) and P (p x p) are invertible and carry the same symmetry, either
% both symmetric or both skew-symmetric. The returned struct supplies the
% canonical-like metric and its inverse, the tangent-space projection, the
% closed-form Euclidean-to-Riemannian gradient and Hessian conversions, the
% Cayley, quasi-geodesic and polar retractions with their second-order
% corrections and matching vector transports, the exponential map, and random
% point and tangent vector generators.

%#ok<*DEFNU>
n = size(Q, 1);
p = size(P, 1);

if issymmetric(Q) && issymmetric(P)
    rho = 1;
elseif issymmetric(Q, "skew") && issymmetric(P, "skew")
    rho = -1;
else
    error('error');
end

invQ = inv(Q);
invP = inv(P);

assert(n >= p, 'The dimension n must be larger than the dimension p.');

if ~exist('beta','var') || isempty(beta)
    beta = 1;
end

guard_reason.cayley_inverse = 'cayley_inverse';
guard_reason.polar_sylvester = 'polar_sylvester';
guard_reason.polar_domain = 'polar_domain';
guard_reason.polar_branch = 'polar_branch';
guard_reason.polar_omega_sylvester = 'polar_omega_sylvester';
transp_guard_log_path = resolve_transp_guard_log_path();
transp_guard_factory_id = register_transp_guard_factory();

M.name = @() sprintf('General Quadratic manifold Gq(%d, %d)', n, p);
M.dim = @() n*p-p*(p+rho)/2;

M.tran = @tran;
    function T = tran(D)
        [n1, n2] = size(D); % n1 >= n2
        if n2 == n
            T = invQ*D'*Q;
        elseif n1 == p
            T = invP*D'*P;
        else
            T = invP*D'*Q;
        end
    end

M.symm = @(D) (D + M.tran(D))/2;
M.skew = @(D) (D - M.tran(D))/2;

I.symm = @(D) (D + D')/2;
I.skew = @(D) (D - D')/2;

M.metric_canonical = @metric_canonical;
    function Me = metric_canonical(X)
        QX = Q*X; XTX = (X'*X);
        Me = beta*(QX*QX') + eye(n) - X/XTX*X';
        Me = I.symm(Me);
    end

M.inv_metric_canonical = @inv_metric_canonical;
    function invMe = inv_metric_canonical(X)
        K = eye(n) - X*M.tran(X);
        invMe = X/(P*P')*X'/beta + K*K';
    end

M.inner = @inner_canonical;
    function inner = inner_canonical(X, D1, D2)
        Me = M.metric_canonical(X);
        inner = trace(D1'*Me*D2);
    end

M.norm = @(X, D) (M.inner(X, D, D))^.5;
M.proj = @(X, U) U - X*M.symm(M.tran(X)*U);
M.tangent = M.proj;
M.tangent2ambient_is_identity = true;
M.tangent2ambient = @(X, U) U;
M.transp_guard_counts = @transp_guard_counts;

% Convert Euclidean gradients to Riemannian gradients
M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        % invM = M.invmetric(X);
        % rgrad = M.proj(X, invQ*invM*egrad*P);
        % rgrad = M.proj(X, (egrad*X'*X+Q*X*egrad'*Q*X)/2);
        invMe = M.inv_metric_canonical(X);
        % invMe = (invMe+invMe')/2;
        rgrad = M.proj(X, invMe*egrad);

    end

% M.ehess2rhess = @ehess2rhess_approx;
M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess_approx(X, egrad, ehess, H)
        invMe = M.inv_metric_canonical(X);
        Y = invMe*egrad;
        Z = H;
        XTXX = (X'*X)\X'; Pi = (eye(n)-X*XTXX);
        R1 = 2*I.symm(beta*Q*X*(Q*Z)'-Pi*Z*XTXX);
        % R2 = Z*M.tran(X)*V+X*M.symm(M.tran(Z)*V);
        R2 = -X*M.symm(M.tran(Z)*Y)-Z*M.symm(M.tran(X)*Y);
        rhess = M.proj(X, invMe*(ehess - R1*Y) +R2);
    end
    % function rhess = ehess2rhess(X, egrad, ehess, H)
    %     invMe = M.inv_metric_canonical(X);
    %     R1 = invMe*ehess;
    %     Y = invMe*egrad; Z = H;
    %     % R2 = -X*M.symm(M.tran(Z)*Y)-Z*M.symm(M.tran(X)*Y);
    %     R2 = -Z*M.symm(M.tran(X)*Y);
    %     XTXX = (X'*X)\X'; Pi = (eye(n)-X*XTXX);
    %     L = (eye(n)-X*M.tran(X)/2); Me = beta*(Q*X)*X'*Q'+Pi;
    %     R3 = 2*I.symm(beta*Q*X*(Q*Z)'-Pi*Z*XTXX);
    %     R3 = invMe*R3*Y;
    %     Y = M.proj(X, Y);
    %     GZ = L*Z*M.tran(X)-X*M.tran(L*Z);
    %     GY = L*Y*M.tran(X)-X*M.tran(L*Y);
    %     DZM = -2*I.symm(Me*GZ-Pi*GZ*Pi)*Y;
    %     DYM = -2*I.symm(Me*GY-Pi*GY*Pi)*Z;
    %     XZY = beta*2*Q'*I.symm(Y*Z')*Q*X-2*Pi*I.symm(Y*Z')*X/(X'*X);
    %     R4 = invMe*(DZM+DYM-XZY);
    %     rhess = M.proj(X, R1 + R2 - R3 + R4);
    % end

    function rhess = ehess2rhess(X, egrad, ehess, H)
        invMe = M.inv_metric_canonical(X);
        R1 = ehess;
        Y = invMe*egrad; Z = H;
        R2 = -Z*M.symm(M.tran(X)*Y);
        XTXX = (X'*X)\X'; Pi = (eye(n)-X*XTXX);
        R3 = 2*I.symm(beta*Q*X*(Q*Z)'-Pi*Z*XTXX)*Y;
        rgrad = M.proj(X, Y);
        R41 = I.symm(beta*Q*X*(Q*Z)'-Pi*Z*XTXX)*rgrad;
        R42 = I.symm(beta*Q*X*(Q*rgrad)'-Pi*rgrad*XTXX)*Z;
        R43 = -beta*Q'*I.symm(rgrad*Z')*Q*X+Pi*I.symm(rgrad*Z')*XTXX';
        % R43 = -tilde{K}(rgrad,Z)/2, so the correction term enters with a plus
        % sign here.
        R4 = R41+R42+R43;
        rhess = M.proj(X, invMe*(R1-R3+R4)+R2);
    end

M.retr_cayley = @retr_cayley;
    function Y = retr_cayley(X, Z, t)

        if nargin == 3
            Z = t*Z;
        end
        L = (eye(n) - X*M.tran(X)/2);
        LZ = L*Z;
        U = [LZ, X]; V = [M.tran(X); -M.tran(LZ)];
        Y = X + U/(eye(2*p)-.5*V*U)*V*X;
    end

M.transp_cayley = @transp_cayley;
    function T = transp_cayley(X, Y, Z)

        % S = eye(p)/(eye(p)+M.tran(X)*Y) + eye(p)/(eye(p)+M.tran(Y)*X) - 2*eye(p);
        % W = (eye(n) + X*M.tran(X))*(2*Y/(eye(p)+M.tran(X)*Y) + X*S);
        Axy = eye(p)+M.tran(X)*Y;
        Ayx = eye(p)+M.tran(Y)*X;
        if ~matrix_guard_accepts(Axy) || ~matrix_guard_accepts(Ayx)
            T = transp_guard_fallback(Y, Z, guard_reason.cayley_inverse);
            return;
        end
        W = 2*Y/Axy +2*X/Ayx - 2*X;
        L = eye(n) - X*M.tran(X)/2;
        UZ = [L*Z, X]; VZ = [M.tran(X); -M.tran(L*Z)];
        UW = [L*W, X]; VW = [M.tran(X); -M.tran(L*W)];
        T1 = eye(2*p)-.5*VW*UW; T2 = UW/T1*VW;
        T = (UZ + .5*T2*UZ)*(VZ*X+.5*VZ*T2*X);
    end

M.retr_pol = @retr_pol;
    function Y = retr_pol(X, Z, t)
        if nargin == 3
            Z = t*Z;
        end
        XZ = X + Z;
        S = sqrtm(M.tran(XZ)*XZ);   % principal sqrt of P^{-1}(X+Z)^T Q (X+Z)
        if norm(imag(S), 'fro') <= 1e-9*max(1, norm(S, 'fro'))
            % The principal real square root exists (true for all sufficiently
            % small steps, including the symmetric indefinite case, where the
            % nonsymmetric argument makes sqrtm emit rounding-level imaginary
            % parts). Discard that rounding noise and retract.
            Y = XZ / real(S);
        else
            % Step too large: P^{-1}(X+Z)^T Q (X+Z) has an eigenvalue on the
            % closed negative real axis, so no real polar factor exists. Return
            % an infeasible point so the line search backtracks to a smaller
            % step where the polar retraction is well defined.
            Y = inf(size(X));
        end
    end
% RetrPolar = @(Z, t) (X+t*Z)/sqrtm(eye(p)+t^2*adjPQ(Z)*Z);

M.transp_pol = @transp_pol;
    function T = transp_pol(X, Y, Z)

        Axy = M.tran(X)*Y;
        Ayx = M.tran(Y)*X;
        if ~sylvester_guard_accepts(Axy, Ayx)
            T = transp_guard_fallback(Y, Z, guard_reason.polar_sylvester);
            return;
        end
        S = lyap(Axy, Ayx, -2*eye(p));
        if ~isfinite_matrix(S)
            T = transp_guard_fallback(Y, Z, guard_reason.polar_sylvester);
            return;
        end
        Zinv = Y*S-X;
        domain_quantity = norm(M.tran(Zinv)*Zinv, 2);
        if ~isfinite(domain_quantity) || domain_quantity >= 1-branch_guard_tol(domain_quantity, 1)
            T = transp_guard_fallback(Y, Z, guard_reason.polar_domain);
            return;
        end
        if ~principal_branch_guard_accepts(S)
            T = transp_guard_fallback(Y, Z, guard_reason.polar_branch);
            return;
        end
        if ~sylvester_guard_accepts(S, S)
            T = transp_guard_fallback(Y, Z, guard_reason.polar_omega_sylvester);
            return;
        end
        Omega = lyap(S, S, M.tran(Z)*Y - M.tran(Y)*Z);
        if ~isfinite_matrix(Omega)
            T = transp_guard_fallback(Y, Z, guard_reason.polar_omega_sylvester);
            return;
        end
        T = Y*Omega+(eye(n) - Y*M.tran(Y))*Z/S;
    end

    function counts = transp_guard_counts()
        global GQ_TRANSP_GUARD_LOG
        if isempty(GQ_TRANSP_GUARD_LOG)
            counts = table();
        else
            counts = struct2table(GQ_TRANSP_GUARD_LOG);
        end
    end

    function T = transp_guard_fallback(Y, Z, reason)
        % Guard fallback: project the incoming vector to T_Y M. This is the
        % paper's tangent projector, so a rejected endpoint inverse still returns
        % a deterministic tangent vector at Y and the optimizer remains feasible.
        record_transp_guard_rejection(reason);
        T = M.proj(Y, Z);
    end

    function ok = matrix_guard_accepts(A)
        ok = isfinite_matrix(A) && rcond(A) > rcond_guard_floor(size(A, 1));
    end

    function ok = sylvester_guard_accepts(A, B)
        dim = size(A, 1);
        Lsylv = kron(eye(dim), A) + kron(B.', eye(dim));
        ok = matrix_guard_accepts(Lsylv);
    end

    function ok = principal_branch_guard_accepts(S)
        lambda = eig(S);
        ok = all(isfinite(lambda)) && min(real(lambda)) > branch_guard_tol(S, p);
    end

    function ok = isfinite_matrix(A)
        ok = all(isfinite(A(:)));
    end

    function tol = rcond_guard_floor(dim)
        % The LU/back-substitution backward error is O(dim*eps). The factor 10
        % leaves a small estimator cushion and rejects only systems whose
        % reciprocal condition estimate is at the roundoff floor.
        tol = 10*max(1, dim)*eps;
    end

    function tol = branch_guard_tol(A, dim)
        tol = 10*max(1, dim)*eps*max(1, norm(A, 1));
    end

    function factory_id = register_transp_guard_factory()
        global GQ_TRANSP_GUARD_LOG
        start_transp_guard_run_if_needed();
        if isempty(GQ_TRANSP_GUARD_LOG)
            factory_id = 1;
        else
            factory_id = max([GQ_TRANSP_GUARD_LOG.factory_id]) + 1;
        end
        row = empty_transp_guard_row();
        row.factory_id = factory_id;
        row.n = n;
        row.p = p;
        row.rho = rho;
        row.beta = beta;
        if isempty(GQ_TRANSP_GUARD_LOG)
            GQ_TRANSP_GUARD_LOG = row;
        else
            GQ_TRANSP_GUARD_LOG(end+1) = row;
        end
        write_transp_guard_log();
    end

    function start_transp_guard_run_if_needed()
        % Each seeded entry-point script begins with `clear`. A normal base-
        % workspace marker is therefore absent on its first factory construction,
        % even though MATLAB global variables survive `clear`. Reset the registry
        % at that boundary so an existing companion CSV is overwritten per run.
        global GQ_TRANSP_GUARD_LOG
        marker_name = 'GQ_TRANSP_GUARD_ACTIVE_RUN';
        marker_exists = evalin('base', ...
            sprintf('exist(''%s'', ''var'') == 1', marker_name));
        if marker_exists
            active_path = evalin('base', marker_name);
            marker_matches = strcmp(string(active_path), transp_guard_log_path);
        else
            marker_matches = false;
        end
        if ~marker_matches
            GQ_TRANSP_GUARD_LOG = repmat(empty_transp_guard_row(), 0, 1);
            assignin('base', marker_name, transp_guard_log_path);
        end
    end

    function record_transp_guard_rejection(reason)
        global GQ_TRANSP_GUARD_LOG
        if isempty(GQ_TRANSP_GUARD_LOG)
            error('Transport guard registry is empty for factory %d.', ...
                transp_guard_factory_id);
        end
        idx = find([GQ_TRANSP_GUARD_LOG.factory_id] == transp_guard_factory_id, 1, 'last');
        if isempty(idx)
            error('Transport guard factory %d is not registered.', ...
                transp_guard_factory_id);
        end
        switch reason
            case guard_reason.cayley_inverse
                GQ_TRANSP_GUARD_LOG(idx).cayley_inverse = GQ_TRANSP_GUARD_LOG(idx).cayley_inverse + 1;
            case guard_reason.polar_sylvester
                GQ_TRANSP_GUARD_LOG(idx).polar_sylvester = GQ_TRANSP_GUARD_LOG(idx).polar_sylvester + 1;
            case guard_reason.polar_domain
                GQ_TRANSP_GUARD_LOG(idx).polar_domain = GQ_TRANSP_GUARD_LOG(idx).polar_domain + 1;
            case guard_reason.polar_branch
                GQ_TRANSP_GUARD_LOG(idx).polar_branch = GQ_TRANSP_GUARD_LOG(idx).polar_branch + 1;
            case guard_reason.polar_omega_sylvester
                GQ_TRANSP_GUARD_LOG(idx).polar_omega_sylvester = GQ_TRANSP_GUARD_LOG(idx).polar_omega_sylvester + 1;
            otherwise
                error('Unknown transport guard reason %s.', reason);
        end
        GQ_TRANSP_GUARD_LOG(idx).total = GQ_TRANSP_GUARD_LOG(idx).total + 1;
        write_transp_guard_log();
    end

    function row = empty_transp_guard_row()
        row = struct('factory_id', 0, 'n', 0, 'p', 0, 'rho', 0, 'beta', 0, ...
            'cayley_inverse', 0, 'polar_sylvester', 0, 'polar_domain', 0, ...
            'polar_branch', 0, 'polar_omega_sylvester', 0, 'total', 0);
    end

    function write_transp_guard_log()
        global GQ_TRANSP_GUARD_LOG
        if strlength(transp_guard_log_path) == 0 || isempty(GQ_TRANSP_GUARD_LOG)
            return;
        end
        writetable(struct2table(GQ_TRANSP_GUARD_LOG), transp_guard_log_path);
    end

    function log_path = resolve_transp_guard_log_path()
        output_dir = fullfile(pwd, 'results');
        if exist(output_dir, 'dir') ~= 7
            log_path = "";
            return;
        end
        caller = 'standalone';
        stack = dbstack('-completenames');
        for istack = 2:numel(stack)
            [~, candidate] = fileparts(stack(istack).file);
            if ismember(candidate, {'example_symm', 'example_skew', ...
                    'example_trace'})
                caller = erase(candidate, {'example_'});
                break;
            end
        end
        log_path = string(fullfile(output_dir, ...
            sprintf('transp_guard_rejections_%s.csv', caller)));
    end

% M.retr_qr = @retr_qr;
%     function Y = retr_qr(X, Z, t)
%         if nargin == 3
%             Z = t*Z;
%         end
%         if rho == +1
%             Y = rootQ\(rootQ*(X+Z)/qr(rootQ*(X+Z),'econ'))*rootP;
%         elseif rho == -1
%             Y = rootQ\MsGS(rootQ*(X+Z)/rootP)*rootP;
%         end
%         % Y = (X+Z)/sqrtm(eye(p) + M.tran(Z)*Z);
%     end

M.retr_qgeo = @retr_qgeo;
    function Y = retr_qgeo(X, Z, t)
        % if M.checkts(X, Z) > 1e-8
            G = (eye(n) - X*M.tran(X)/2)*Z;
            Gamma = G*M.tran(X) - X*M.tran(G);
            Z = Gamma*X;
        % end

        if nargin == 3
            Z = t*Z;
        end

        % A1 = M.tran(X)*Z;
        % A4 = (eye(n) - X*M.tran(X))*Z;
        % TT = expm([A1,-M.tran(A4);A4,zeros(n)]);
        % Y = [X, eye(n)]*TT(:,1:p);
        A1 = M.tran(X)*Z;
        A2 = M.tran(Z)*Z;
        TT = expm([A1,-A2;eye(p),A1]);
        Y = [X, Z]*TT(:,1:p)*expm(-A1);

        % Y = @(Z, t) [X, Z]*expm(t*[A1(Z), -A2(Z); eye(p), A1(Z)])...
        %     *[eye(p); zeros(p,p)]*expm(-t*A1(Z));
    end

M.transp_qgeo = @transp_qgeo;
    function T = transp_qgeo(X, Y, Z)
        if M.checkts(X, Z) > 1e-8
            G = (eye(n) - X*M.tran(X)/2)*Z;
            Gamma = G*M.tran(X) - X*M.tran(G);
            Z = Gamma*X;
        end
        
        A4 = eye(n) - X*M.tran(X);
        TM = M.tran(X)*Y;
        TN = A4*Y;
        T = Z*TM - X*(M.tran(Z)*TN);
    end
% =====================================================================
%  Christoffel operator of g^beta, and the second-order corrections of
%  the three retractions.
%
%  The proof of the Riemannian-Hessian proposition already carries the
%  Levi-Civita connection of g^beta, although it is not labelled as such.
%  Writing hess f(X)[Z] = nabla_Z grad f and matching the four
%  contributions I_1..I_4 term by term gives, for a tangent field Y,
%
%      nabla_Z Y = P_T( D_Z Y + (M_X^beta)^{-1} K(Z,Y) ),
%
%  so the Christoffel operator of g^beta in the ambient chart of
%  R^{n x p} is
%
%      Gamma^beta_X(Z,U) = P_T( (M_X^beta)^{-1} K(Z,U) ),
%
%  with K the metric-compatibility correction
%      K(Z,U) = ( D_Z M^beta . U + D_U M^beta . Z - Ktilde(Z,U) )/2
%  of that same proof. D_Z M^beta, Ktilde and P_T below are transcribed
%  from the manuscript displays in exactly the form the R41/R42/R43
%  blocks of ehess2rhess above already use, so the two agree by
%  construction and not by coincidence.
%
%  The covariant acceleration at t = 0 of the curve c(t) = R_X(tZ) is
%
%      alpha_R(Z) = P_T(c''(0)) + Gamma^beta_X(Z,Z),
%
%  and the second-order correction of R is Rtilde_X(Z) = R_X(Z -
%  alpha_R(Z)/2). alpha_R is homogeneous of degree two in Z, so
%  Rtilde_X(tZ) = R_X(tZ - (t^2/2) alpha_R(Z)) and the three retr2_*
%  fields take (X, Z, t) exactly as the first-order retr_* fields do.
%
%  Ambient accelerations, all three read off the closed forms of Sec. 4
%  with S = X^{(P,Q)}Z (P-skew) and N = Z^{(P,Q)}Z (P-symmetric):
%
%      quasi-geodesic  c''(0) = -X N,              P_T(c''(0)) = 0,
%      polar           c''(0) = -X N,              P_T(c''(0)) = 0,
%      Cayley          c''(0) = Z S - X S^2 - X N, P_T(c''(0)) = Z S - X S^2,
%
%  since X N and X S^2 lie in the normal space {X S : S in S_sym(P)}.
%
%  Cost. Transcribed literally these expressions form n-by-n matrices,
%  i.e. O(n^3) per call. Reassociating each of the six terms of K brings
%  the cost to O(n^2 p + n p^2 + p^3), the order of the retractions
%  themselves. The fields below are diagnostics rather than inner-loop
%  code, so they follow the manuscript displays literally.
% =====================================================================

M.christoffel = @christoffel;
    function G = christoffel(X, Z, U)
        % Gamma^beta_X(Z,U) = P_T( (M_X^beta)^{-1} K(Z,U) ). Symmetric in
        % (Z,U); U defaults to Z, which is the case the second-order
        % correction needs.
        if nargin < 3 || isempty(U)
            U = Z;
        end
        XTXX = (X'*X)\X';            % X^dagger
        Pi   = eye(n) - X*XTXX;
        DZM_U = 2*I.symm(beta*Q*X*(Q*Z)' - Pi*Z*XTXX)*U;   % D_Z M^beta . U
        DUM_Z = 2*I.symm(beta*Q*X*(Q*U)' - Pi*U*XTXX)*Z;   % D_U M^beta . Z
        Ktil  = 2*beta*Q'*I.symm(U*Z')*Q*X - 2*Pi*I.symm(U*Z')*XTXX';
        Kcal  = (DZM_U + DUM_Z - Ktil)/2;
        G = M.proj(X, M.inv_metric_canonical(X)*Kcal);
    end

M.retr_accel = @retr_accel;
    function Acc = retr_accel(X, Z, which)
        % Tangential part P_T(c''(0)) of the ambient acceleration of
        % c(t) = R_X(tZ), for the retraction named by WHICH.
        switch which
            case 'cay'
                S = M.tran(X)*Z;
                Acc = M.proj(X, Z*S - X*(S*S));
            case {'qgeo', 'pol'}
                Acc = zeros(n, p);
            otherwise
                error('generalfactory:retr_accel', ...
                    'Unknown retraction key ''%s''.', which);
        end
    end

M.retr2_alpha = @retr2_alpha;
    function al = retr2_alpha(X, Z, which)
        % Covariant acceleration alpha_R(Z) of c(t) = R_X(tZ) at t = 0.
        al = M.retr_accel(X, Z, which) + M.christoffel(X, Z, Z);
    end

M.retr2_cay = @retr2_cay;
    function Y = retr2_cay(X, Z, t)
        if nargin == 3
            Z = t*Z;
        end
        Y = M.retr_cayley(X, Z - M.retr2_alpha(X, Z, 'cay')/2);
    end

M.retr2_qgeo = @retr2_qgeo;
    function Y = retr2_qgeo(X, Z, t)
        if nargin == 3
            Z = t*Z;
        end
        Y = M.retr_qgeo(X, Z - M.retr2_alpha(X, Z, 'qgeo')/2);
    end

M.retr2_pol = @retr2_pol;
    function Y = retr2_pol(X, Z, t)
        if nargin == 3
            Z = t*Z;
        end
        Y = M.retr_pol(X, Z - M.retr2_alpha(X, Z, 'pol')/2);
    end
% NOTE: M.retr2 is deliberately NOT set. Manopt's checkhessian.m prefers
% M.exp, then M.retr2, then M.retr as its stepper, so defining M.retr2
% here would silently change which curve a script steps along once it
% strips M.exp. The three second-order fields above are addressed by name
% instead.

M.retr = M.retr_cayley;
M.transp = M.transp_cayley;
M.selecttransp = @selecttransp; % M.transp(x, newx, grad)
    function transp = selecttransp()
        if isequal(M.retr, M.retr_cayley)
            transp = M.transp_cayley;
        elseif isequal(M.retr, M.retr_pol)
            transp = M.transp_pol;
        elseif isequal(M.retr, M.retr_qgeo)
            transp = @(x1, x2, d) M.proj(x2, d);
            % transp = M.transp_qgeo;
        elseif isequal(M.retr, M.retr_qr)
            % transp = M.transp_qr;
        else
            error('error');
        end
    end

M.feasi = @feasi;
    function res = feasi(X)
        res = norm(M.tran(X) * X - eye(p));
    end

M.checkts = @(X,Z) norm(M.tran(X)*Z +M.tran(Z)*X,'fro');

M.hash = @(X) ['z' hashmd5(X(:))];

M.exp = @exponential;
    function Y = exponential(X, U, t)
        if nargin == 2
            tU = U;
        else
            tU = t*U;
        end
        A = M.tran(X)*tU;
        B = (eye(n)-X*M.tran(X))*tU;
        Y = [X, eye(n)] * expm([2*A, -M.tran(B); B, zeros(n)])...
            * [eye(p); zeros(n, p)] * expm(-A);
    end

M.rand = @random;
    function X = random()
        if rho == 1
            [PhiP, lamP] = eig(P);
            [PhiQ, lamQ] = eig(Q);
            % Inertia-matched selection: pair every eigendirection of P with an
            % eigendirection of Q of the SAME sign. Otherwise lamP/lamQ(iid,iid)
            % has negative diagonal entries (when P or Q is indefinite, e.g. the
            % indefinite Stiefel manifold), so sqrtm returns a complex factor and
            % X becomes complex. For SPD P,Q this reduces to the previous code.
            dP = diag(lamP); dQ = diag(lamQ);
            posQ = find(dQ > 0); negQ = find(dQ < 0);
            posQ = posQ(randperm(numel(posQ)));
            negQ = negQ(randperm(numel(negQ)));
            assert(numel(posQ) >= nnz(dP > 0) && numel(negQ) >= nnz(dP < 0), ...
                ['Gq(P,Q) is empty: the inertia of P must be dominated by that ' ...
                 'of Q, i.e. i_+(P) <= i_+(Q) and i_-(P) <= i_-(Q).']);
            iid = zeros(1, p); kp = 1; km = 1;
            for j = 1:p
                if dP(j) > 0
                    iid(j) = posQ(kp); kp = kp + 1;
                else
                    iid(j) = negQ(km); km = km + 1;
                end
            end
        else
            [PhiP, lamP] = YoulaDecomposition(P);
            [PhiQ, lamQ] = YoulaDecomposition(Q);
            % Enforce a consistent sign convention (positive frequency) in every
            % 2-by-2 Youla block of P and Q. Otherwise a matched pair of blocks
            % with opposite-sign frequencies makes lamP/lamQ(iid,iid) indefinite,
            % so sqrtm returns a complex factor and X becomes complex.
            [PhiP, lamP] = posfreq(PhiP, lamP);
            [PhiQ, lamQ] = posfreq(PhiQ, lamQ);
            iid = randperm(n/2, p/2);
            iid = sort([2*iid-1, 2*iid]);
        end
        % iid = 1:p;
        S = sqrtm(lamP/lamQ(iid,iid));
        S = real(S);   % argument is SPD after matching; strip rounding-level imag part
        X = PhiQ(:, iid)*S*PhiP';
        X = X*expm(M.skew(randn(p, p)));
    end

    function [Phi, Lam] = posfreq(Phi, Lam)
        % Normalize each 2-by-2 Youla block [[0, b], [-b, 0]] to have b > 0 by
        % swapping the two basis vectors of the block whenever b < 0.
        m = size(Lam, 1);
        for b = 1:2:m-1
            if Lam(b, b+1) < 0
                Phi(:, [b, b+1]) = Phi(:, [b+1, b]);
                Lam([b, b+1], :) = Lam([b+1, b], :);
                Lam(:, [b, b+1]) = Lam(:, [b+1, b]);
            end
        end
    end

M.randvec = @generatetangent;
    function U = generatetangent(X)
        U = M.skew(randn(n))*X;
        U = U / norm(U(:));
    end

M.lincomb = @matrixlincomb;

M.zerovec = @(x) zeros(n, p);

M.vec = @(x, u_mat) u_mat(:);
M.mat = @(x, u_vec) reshape(u_vec, [n, p]);
% M.vec is plain Frobenius vectorisation, but this manifold's metric is the
% non-Frobenius canonical-like M_X = beta*(QX)(QX)' + I - X(X'X)^{-1}X'.
% vec/mat are therefore NOT isometries and must not claim to be: Manopt feeds
% this flag straight into eigs_opts.issym in hessianspectrum, i.e. it asserts
% to ARPACK that the operator is symmetric in the vectorised coordinates.
M.vecmatareisometries = @() false;

end
