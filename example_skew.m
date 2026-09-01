% example_skew.m
%
% Preconditioning experiment, skew-symmetric case. On the same four quadratic
% objectives (eigenvalue, Brockett, tridiagonal, quadratic) the baseline
% symplectic Stiefel manifold Sp(p,n) is compared with the reshaped constraint
% Gq(P,Q) under a balanced and under an optimally designed choice of the
% constraint spectra. Each run records the Riemannian gradient norm along the
% iterate sequence and the condition number of the Riemannian Hessian at the
% converged point.
%
% Reproduces the skew-symmetric half of Section 6.1 of the paper
% "Preconditioning for Ill-Conditioned Quadratic Optimization via Reshaping
% Constraints".
%
% Seed rng(0), set once at the top. Dimensions n = 100, p = 20. The Hessian
% spectra dominate the cost of the run.
%
% Requires Manopt and CVX. CVX solves the linear program that designs the
% constraint spectra. The baseline manifold is symplecticfactory.m and the
% skew-symmetric eigendecompositions use YoulaDecomposition.m.
%
% Run from this directory:
%   >> example_skew
%
% Outputs, all written to results/ , with <problem> ranging over eigenvalue,
% brockett, tridiagonal and quadratic:
%   skew_<problem>_n_100_p_20.pdf   convergence figure, one per objective
%   skew_<problem>_n_100_p_20.csv   the matching table

clear; close all; clc; rng(0);

%% Create the manifold
% n = 20; p = 10;
n = 100; p = 20;
J = @(m) [zeros(m/2) eye(m/2);-eye(m/2) zeros(m/2)];
Jn = J(n); Jp = J(p);
symm = @(D) (D+D')/2;
skew = @(D) (D-D')/2;

C = gallery('lehmer', n);
% Jp*S is Hamiltonian for every symmetric S, so expm(Jp*S) is symplectic. S must
% also be positive definite: then Jp*S has purely imaginary spectrum and expm is
% bounded. An indefinite S (e.g. W+W') makes Jp*S hyperbolic, and at p = 20 the
% draw reaches norm 4.3e2 with condition number 1.8e5, which leaves the converged
% iterate feasible only to 1e-8 and the structural Hessian zeros unresolvable.
W = randn(p,p);
E = expm(Jp*(W'*W + eye(p)/10));
Y0 = [E(1:p/2,:);zeros((n-p)/2,p);E(p/2+1:end,:);zeros((n-p)/2,p)];

%% Create the problem
prob{1}.D = eye(p);
prob{1}.name = 'eigenvalue';

prob{2}.D = diag(1:p);
prob{2}.name = 'brockett';

e = ones(p, 1);
prob{3}.D = full(spdiags([-e 2*e -e], -1:1, p, p));
prob{3}.name = 'tridiagonal';

prob{4}.D = gallery('lehmer', p);
prob{4}.name = 'quadratic';

lenp = numel(prob);

line_list = {'-', '--', '-.', ':', '-.'};
color_list = {'k', 'r', 'b', 'g', 'm'};
marker_list = {'o', 's', '^', 'd', 'x'};
FONT_NAME = 'Times New Roman';
FONT_SIZE_LABEL = 9;
FONT_SIZE_TICK = 8;
FONT_SIZE_LEGEND = 9;
LINE_WIDTH = 2;
AXIS_LINE_WIDTH = 1.0;
MARKER_SIZE = 4;
MARKER_NUM = 5;
FIG_WIDTH = 7;
FIG_HEIGHT = 2;

left = 0.05;    % minimum left margin (maximizes the subplots)
right = 0.02;   % minimum right margin
bottom = 0.2;  % minimum bottom margin
top = 0.12;     % top margin, leaving room for the title
hspace = 0.06;  % horizontal gap between subplots
vspace = 0.01;  % vertical gap between subplots
ncols = 3; nrows = 1;
subplotW = (1 - left - right - (ncols-1)*hspace) / ncols;
subplotH = (1 - top - bottom - (nrows-1)*vspace) / nrows;

posx = @(ic) left + (ic-1)*(subplotW + hspace);
posy = 1 - top - 1*subplotH;

output_dir = 'results/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for ip = 1:lenp
    D = prob{ip}.D;
    filename = sprintf('skew_%s_n_%d_p_%d', prob{ip}.name, n, p);
    disp(filename);

    wC = sympl_eig(C, n/2);                 % n symplectic eigenvalues of C, desc
    wD = sympl_eig(D, p/2);                 % p symplectic eigenvalues of D, desc
    c  = sort(1 ./ wC, 'descend');        % reciprocal symplectic eigs of C, desc
    d  = sort(wD,      'descend');        % symplectic eigs of D,            desc

    %% Create the solver
    sol{1}.M = symplecticfactory(n/2, p/2, 'canonical');
    sol{1}.cost = @(Y) trace(Y'*C*Y*D);
    sol{1}.egrad = @(Y) 2*C*Y*D;
    sol{1}.ehess = @(Y, Ydot) 2*C*Ydot*D;
    sol{1}.Y0 = Y0;
    % feasi = ||X^{(P,Q)}X - I_p||_F = ||P^{-1}(X'QX - P)||_F; here P = Jp, Q = Jn.
    sol{1}.feasi = @(Y) norm(Jp\(Y'*Jn*Y - Jp), 'fro');
    sol{1}.name = "Sp(p,n)";
    sol{1}.condhat = nan;

    % A = I, B = I
    tildeQ = inv(sqrtm(C));
    tildeP = sqrtm(D);
    Q = skew(tildeQ'*Jn*tildeQ);
    P = skew(tildeP'*Jp*tildeP);
    A = symm(tildeQ'*C*tildeQ);
    B = symm(tildeP\D/tildeP');
    sol{2}.M = generalfactory(Q, P);
    sol{2}.cost = @(Y) trace(Y'*A*Y*B);
    sol{2}.egrad = @(Y) 2*A*Y*B;
    sol{2}.ehess = @(Y, Ydot) 2*A*Ydot*B;
    sol{2}.Y0 = tildeQ\Y0*tildeP;
    sol{2}.feasi = @(Y) norm(P\(Y'*Q*Y - P), 'fro');
    sol{2}.name = "Gq(P,Q)";
    sol{2}.condhat = GQ_spectrum(c, d, d, c, n, p);

    % EQopt + EPopt cvx
    rootC = sqrtm(C);
    rootD = sqrtm(D);
    frakC = rootC\Jn/rootC;
    frakD = rootD*Jp*rootD;
    [UC, dEC] = YoulaDecomposition(frakC);
    [UD, dED] = YoulaDecomposition(frakD);
    EC = diag(dEC(1:2:n, (1:2:n)+1));
    ED = diag(dED(1:2:p, (1:2:p)+1));

    [Iidx1, Jidx1] = meshgrid(1:p/2, 1:p/2);
    Mminus = ((1./EC(Iidx1)-1./EC(Jidx1)).*(ED(Jidx1)-ED(Iidx1)))/2;
    logMminus = log(Mminus);
    Mplus = ((1./EC(Iidx1)+1./EC(Jidx1)).*(ED(Jidx1)+ED(Iidx1)))/2;
    logMplus = log(Mplus);
    % Diagonal ell_i^0 rows of (LP^-), in the same one-half normalization
    % used by Mplus and Nplus. The common normalization does not affect U-L.
    logMzero = log(2 .* ED ./ EC(1:p/2));
    
    [Iidx2, Jidx2] = meshgrid(1:p/2, (p/2+1):n/2);
    
    Nminus = (1./EC(Jidx2) - 1./EC(Iidx2)) .* ED(Iidx2);
    logNminus = log(Nminus);
    Nplus = (1./EC(Jidx2) + 1./EC(Iidx2)) .* ED(Iidx2);
    logNplus = log(Nplus);
    
    [~, ~, idxED] = unique(ED, 'stable');
    
    cvx_begin quiet
        variables x(p/2) y(n/2-p/2) U L
        eq_mask = idxED == idxED';
        [i,j] = find(triu(eq_mask, 1));
        x(i) == x(j);
        
        minimize(U - L)
        
        idxM = triu(logMminus > -inf, 1);
        [iM,jM] = find(idxM);
        varM = logMminus(idxM) - x(iM) - x(jM);
        varM <= U;
        varM >= L;
    
        idxM = triu(logMplus > -inf, 1);
        [iM,jM] = find(idxM);
        varM = logMplus(idxM) - x(iM) - x(jM);
        varM <= U;
        varM >= L;

        varMzero = logMzero - 2 .* x;
        varMzero <= U;
        varMzero >= L;
        
        idxN = logNminus > -inf;
        [iN,jN] = find(idxN);
        varN = logNminus(idxN) + y(iN) - x(jN);
        varN <= U;
        varN >= L;
    
        idxN = logNplus > -inf;
        [iN,jN] = find(idxN);
        varN = logNplus(idxN) + y(iN) - x(jN);
        varN <= U;
        varN >= L;
    cvx_end
    opt_lp = cvx_optval;

    % Lexicographic tie-break: stage two is posed as close to the optimal face
    % as the solver can certify. The ladder starts at solver precision and
    % escalates only if CVX does not return 'Solved'; the realized slack is
    % printed so the reported spread's distance from opt_lp is auditable.
    for slack_scale = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        slack = slack_scale * max(1.0, abs(opt_lp));
        cvx_begin quiet
            variables x(p/2) y(n/2-p/2) U L
            eq_mask = idxED == idxED';
            [i,j] = find(triu(eq_mask, 1));
            x(i) == x(j);

            minimize(sum_square(x) + sum_square(y))

            U - L <= opt_lp + slack;

            idxM = triu(logMminus > -inf, 1);
            [iM,jM] = find(idxM);
            varM = logMminus(idxM) - x(iM) - x(jM);
            varM <= U;
            varM >= L;

            idxM = triu(logMplus > -inf, 1);
            [iM,jM] = find(idxM);
            varM = logMplus(idxM) - x(iM) - x(jM);
            varM <= U;
            varM >= L;

            varMzero = logMzero - 2 .* x;
            varMzero <= U;
            varMzero >= L;

            idxN = logNminus > -inf;
            [iN,jN] = find(idxN);
            varN = logNminus(idxN) + y(iN) - x(jN);
            varN <= U;
            varN >= L;

            idxN = logNplus > -inf;
            [iN,jN] = find(idxN);
            varN = logNplus(idxN) + y(iN) - x(jN);
            varN <= U;
            varN >= L;
        cvx_end
        if strcmp(cvx_status, 'Solved')
            break;
        end
    end
    if ~strcmp(cvx_status, 'Solved')
        error('Stage-2 CVX solve failed at maximum relative slack 1e-4: %s', cvx_status);
    end
    disp(['Stage-2 slack: ', num2str(slack, 17)]);
    disp(['cvx opt: ', num2str(exp(U-L))]);
    
    UQ = orth(rand(n));
    EQ = [ones(p/2,1); exp(y)];
    Q = skew(UQ*kron(diag(EQ),J(2))*UQ'); 
    
    UP = orth(rand(p));
    EP = exp(x);
    P = skew(UP*kron(diag(EP),J(2))*UP'); 
    
    breveQ = rootC\UC*kron(diag(EQ./EC).^.5 ,eye(2))*UQ';
    breveP = rootD*UD*kron(diag(EP./ED).^.5 ,eye(2))*UP';
    
    A = symm(breveQ'*C*breveQ); 
    B = symm(breveP\D/breveP');

    sol{3}.M = generalfactory(Q, P);
    sol{3}.cost = @(Y) trace(Y'*A*Y*B);
    sol{3}.egrad = @(Y) 2*A*Y*B;
    sol{3}.ehess = @(Y, Ydot) 2*A*Ydot*B;
    sol{3}.Y0 = breveQ\Y0*breveP;
    sol{3}.feasi = @(Y) norm(P\(Y'*Q*Y - P), 'fro');
    sol{3}.name = "Gq(P,Q)";
    sol{3}.condhat = GQ_spectrum(c, d, EP, EQ, n, p);

    fstar = 2*sum( 1./c(1:p/2) .* d(1:p/2));
    disp(fstar)

    opt.verbosity = 0;
    opt.maxiter = 1e4;
    opt.tolgradnorm = 1e-6;   % set explicitly (equals the Manopt default) for reproducibility
    lens = numel(sol);
    out_infos = cell(lens, 1);
    out_lambdas = cell(lens, 1);
    % out_conds = nan(lens, 1);
    condstr = cell(lens, 1);
    % Semantic method labels for the CSV row names and the figure legend.
    % Order = sol{1..3}.
    method_labels = {'Sp', 'PC-bal', 'PC-opt'};
    assert(numel(method_labels) == lens, 'method_labels count must equal lens');
    T_vals = cell(lens, 1);
    for is = 1:lens
        solver = sol{is};
        % [x, cost, info, ~] = trustregions(solver, solver.Y0, opt);
        [x, cost, info, ~] = conjugategradient(solver, solver.Y0, opt);
        [cond, lambda, zinfo] = getcond(solver, x);
        fprintf('zero-modes %-8s tol=%.3e kept_min=%.6e rejected_max=%.3e n_rejected=%d\n', ...
            method_labels{is}, zinfo.zero_tol, zinfo.kept_min, zinfo.rejected_max, zinfo.n_rejected);
        out_infos{is} = info;
        out_lambdas{is} = lambda;
        condstr{is} = method_labels{is};
        % condstr{is} = sprintf('method %d ($\\kappa=%.2e$)', is, cond);
        T_vals{is} = [solver.cost(x), info(end).gradnorm, ...
            solver.feasi(x), info(end).iter, info(end).time,...
            info(end).time/info(end).iter, cond, solver.condhat];
    end

    %%
    varNames = ["fval", "gradf", "feasi", "iter", "time", "time_per_iter", "cond", "cond hat"];
    tablerow = permute(T_vals, [2,1]);
    tablerow = reshape(tablerow, [], 1);
    T = array2table(vertcat(tablerow{:}));
    T.Properties.VariableNames = varNames;
    T.Properties.RowNames = method_labels(:);
    disp(T);
    %%
    close all;
    fig = figure('Color','white','Units','inches', 'Position',[0 0 FIG_WIDTH FIG_HEIGHT]);
    set(fig, 'PaperUnits','inches','PaperSize',[FIG_WIDTH FIG_HEIGHT],...
        'PaperPosition',[0 0 FIG_WIDTH FIG_HEIGHT]);
    ax = gobjects(3, 1);

    ax(1) = axes('Units','normalized','Position', [posx(1), posy, subplotW, subplotH]);
    hold on; box on; grid off;

    for is = 1:lens
        out_info = out_infos{is};
        marker_idx = round(logspace(0, log10(length([out_info.time])), MARKER_NUM));
        % marker_idx = round(linspace(1, length([out_info.time]), MARKER_NUM));
        semilogy([out_info.time], [out_info.cost]-fstar, 'LineStyle', line_list{is},...
            'Color', color_list{is}, 'Marker', marker_list{is}, ...
            'LineWidth', LINE_WIDTH, 'MarkerSize', MARKER_SIZE, ...
            'MarkerIndices', marker_idx);
        % hold on; box on; grid off;
    end
    set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize',FONT_SIZE_TICK, ...
        'FontName', FONT_NAME, 'LineWidth', AXIS_LINE_WIDTH);
    xlabel('Time [s]', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
    ylabel('Cost value gap', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);

    ax(2) = axes('Units','normalized','Position', [posx(2), posy, subplotW, subplotH]);
    hold on; box on; grid off;
    for is = 1:lens
        out_info = out_infos{is};
        marker_idx = round(logspace(0, log10(length([out_info.iter])), MARKER_NUM));
        % marker_idx = round(linspace(1, length([out_info.iter]), MARKER_NUM));
        semilogy([out_info.iter], [out_info.gradnorm], 'LineStyle', line_list{is},...
            'Color', color_list{is}, 'Marker', marker_list{is}, ...
            'LineWidth', LINE_WIDTH, 'MarkerSize', MARKER_SIZE, ...
            'MarkerIndices', marker_idx);
        hold on; box on; grid off;
    end
    set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize',FONT_SIZE_TICK, ...
        'FontName', FONT_NAME, 'LineWidth', AXIS_LINE_WIDTH);
    xlabel('Iteration #', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
    ylabel('Gradient norm', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);

    ax(3) = axes('Units','normalized','Position', [posx(3), posy, subplotW, subplotH]);
    hold on; box on; grid off;
    for is = 1:lens
        out_lambda = sort(out_lambdas{is});
        marker_idx = round(logspace(0, log10(length(out_lambda)), MARKER_NUM));
        stairs(out_lambda, 'LineStyle', line_list{is},...
            'Color', color_list{is}, 'LineWidth', LINE_WIDTH);
        lambdax = 1:length(out_lambda);
        hold on; box on; grid off;
        plot(lambdax(marker_idx), out_lambda(marker_idx), 'LineStyle', 'none',...
            'Color', color_list{is}, 'Marker', marker_list{is}, ...
            'LineWidth', LINE_WIDTH, 'MarkerSize', MARKER_SIZE);
    end
    set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize',FONT_SIZE_TICK, ...
        'FontName', FONT_NAME, 'LineWidth', AXIS_LINE_WIDTH);
    xlabel('Eigenvalue number (sorted)', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
    ylabel('Eigenvalues of Hessian', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);

    for iax = 1:numel(ax)
        pos = get(ax(iax), 'Position');
        pos(2) = pos(2) + 0.08;
        set(ax(iax), 'Position', pos);
    end

    leg_ax = axes('Position', [0.2, 0, 0.6, 0.04], 'Visible', 'off');
    hold on;
    for is = 1:lens
        plot(NaN, NaN, 'LineStyle', line_list{is},...
            'Color', color_list{is}, 'Marker', marker_list{is}, ...
            'LineWidth', LINE_WIDTH, 'MarkerSize', MARKER_SIZE);
    end
    legend(leg_ax, condstr, 'Position', [0.2, 0.02, 0.6, 0.06], ...
        'Orientation', 'horizontal', 'Box', 'on', 'FontName', FONT_NAME, ...
        'FontSize', FONT_SIZE_LEGEND, 'Interpreter', 'latex');

    filename_csv = fullfile(output_dir, sprintf('%s.csv', filename));
    filename_pdf = fullfile(output_dir, sprintf('%s.pdf', filename));

    writetable(T, filename_csv, 'WriteRowNames', true);
    exportgraphics(fig, filename_pdf, 'ContentType', 'vector', ...
        'BackgroundColor', 'none', 'Resolution', 300);
end

%%
% ======================================================================
% Symplectic eigenvalues of an SPD matrix A (size 2m x 2m) via Williamson:
%   spec(J_{2m} A) = { +/- i w_k },  w_k > 0  the symplectic eigenvalues.
%   Returns the m values w_k sorted DESCENDING.
% ======================================================================
function w = sympl_eig(A, m)
    J = [zeros(m,m) eye(m); -eye(m) zeros(m,m)];
    ev = eig(J * A);
    w  = sort(imag(ev(imag(ev) > 1e-12)), 'descend');   % m positive halves
end

% ======================================================================
% Condition number of the Riemannian Hessian via Manopt's hessianspectrum.
% ======================================================================
function [cond, lambda, zinfo] = getcond(solver, x)
    lambda = hessianspectrum(solver, x);
    % Retain genuine small positive modes. A fixed 1e-5 cutoff deleted the
    % true 8.93e-6 minimum mode in the skew tridiagonal PC-bal experiment.
    zero_tol = 1e-12 * max([1; abs(lambda(:))]);
    lambda = real(lambda);
    rejected = lambda(lambda <= zero_tol);
    lambda = lambda(lambda > zero_tol);
    lambda = sort(lambda, 'descend');
    cond = lambda(1) / lambda(end);
    % Archive the classification boundary, so a reader can check that no
    % genuine mode was discarded.
    zinfo = struct('zero_tol', zero_tol, 'kept_min', lambda(end), ...
                   'rejected_max', maxrej(rejected), 'n_rejected', numel(rejected));
end

function m = maxrej(r)
    if isempty(r), m = NaN; else, m = max(r); end
end

% ======================================================================
% Closed-form Gq(P, Q) Hessian spectrum at the optimum.
% ======================================================================
function s = GQ_spectrum(c, d, x, y, n, p, beta)
    if nargin < 7 || isempty(beta), beta = 1; end

    % ---- half sizes ----
    if mod(n,2) ~= 0 || mod(p,2) ~= 0
        error('sympl_spectrum:evenDims', 'n and p must be even for the skew-symmetric (symplectic) case.');
    end
    np_ = n/2;   pp_ = p/2;

    % ---- shape / size checks ----
    c = c(:);  d = d(:);  x = x(:);  y = y(:);

    ib = 1/beta;
    vals = [];   % distinct nonzero eigenvalues (before applying multiplicity 2)

    % ---- internal block:  1 <= i < j <= p/2   (H_f1, carries 1/beta) ----
    % (1/beta)/(x_i x_j) * [ (d_i/c_j + d_j/c_i) +/- (d_i/c_i + d_j/c_j) ]
    if pp_ >= 2
        [I, Jj] = find(triu(true(pp_), 1));     % all pairs i<j over 1..p/2
        i = I;  j = Jj;
        base = ib ./ (x(i).*x(j));
        s1 = d(i)./c(j) + d(j)./c(i);
        s2 = d(i)./c(i) + d(j)./c(j);
        vals = [vals; base.*(s1 + s2); base.*(s1 - s2)];   %#ok<AGROW>
    end

    % ---- diagonal block:  1 <= i <= p/2   (H_f1, carries 1/beta) ----
    % (1/beta) * 4 d_i / (x_i^2 c_i)
    ii = (1:pp_).';
    vals = [vals; ib .* 4.*d(ii) ./ (x(ii).^2 .* c(ii))];

    % ---- external block:  1 <= i <= p/2 < j <= n/2   (H_f2, NO 1/beta) ----
    % 2 d_i y_j / x_i * ( 1/c_j +/- 1/c_i )
    if np_ > pp_
        [Ig, Jg] = ndgrid(1:pp_, (pp_+1):np_);  % i over engaged, j over complementary
        i = Ig(:);  j = Jg(:);
        pre = 2 .* d(i) .* y(j) ./ x(i);
        vals = [vals; pre.*(1./c(j) + 1./c(i)); pre.*(1./c(j) - 1./c(i))];   %#ok<AGROW>
    end

    % ---- drop numerical zeros, then apply multiplicity 2 ----
    tol  = 1e-12 * max([1; abs(vals)]);
    vals = vals(abs(vals) > tol);
    nnz_expected = n*p - p^2/2;            % total nonzero count (with multiplicity 2)
    if 2*numel(vals) ~= nnz_expected
        warning('sympl_spectrum:count', ...
            'Got %d nonzero eigenvalues (x2), expected %d; check for degenerate spectra.', ...
            2*numel(vals), nnz_expected);
    end

    % ---- assemble full multiset: doubled nonzeros + p^2/2 structural zeros ----
    nz = repelem(vals, 2);              % multiplicity 2
    s  = condfromspec(nz);
end


% ======================================================================
% Condition number from a closed-form spectrum (drop structural zeros).
% ======================================================================
function k = condfromspec(s)
    s = s(abs(s) > 1e-12);
    k = max(s)/min(s);
end
