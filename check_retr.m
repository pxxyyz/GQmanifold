% check_retr.m
%
% Taylor-remainder check of the closed-form Riemannian gradient and Hessian of
% Gq(P,Q) under the beta-metric (beta = 1), on the trace objective
%
%     f(X) = tr(X'*A*X),   egrad f = 2*A*X,   ehess f[Xdot] = 2*A*Xdot,
%
% at a random feasible point X0, for the three retractions of the paper and for
% the three configurations of (P,Q): symmetric, skew-symmetric, and symmetric
% indefinite. Three quantities per column, all on Manopt's own step grid
% t = logspace(-8, 0, 51) and with Manopt's own sliding-window log-log fit:
%
%   row 1  E(t)   = |f(R_X(tZ)) - f(X) - t g(grad f, Z)|            under R
%   row 2  E_H(t) = |E-model minus (t^2/2) g(hess f[Z], Z)|         under R
%   row 3  the same E_H(t)                                          under Rtilde
%
% Here R is one of the first-order retractions R^cay, R^qgeo, R^pol and Rtilde
% is its second-order correction Rtilde_X(Z) = R_X(Z - alpha_R(Z)/2), where
% alpha_R(Z) = P_T(c''(0)) + P_T((M_X^beta)^{-1} K(Z,Z)) is the covariant
% acceleration of t -> R_X(tZ) at t = 0.
%
% Row 2 is O(t^2) because none of the three retractions is second order for
% g^beta. Row 3 is O(t^3), which is the statement that the correction works,
% and it is the only row of the check that exercises the metric-compatibility
% correction K of the Hessian formula: K enters the Hessian multiplied by
% grad f(X), so it is invisible at a critical point.
%
% Reproduces the Taylor-remainder figure of Section 5.1, "Checking the gradient
% and Hessian", of the paper "Generalized Quadratic Matrix Manifolds: Geometry
% and Computations".
%
% Seed rng(1), set once at the top. Dimensions n = 100, p = 10.
%
% Run from this directory, with Manopt on the path:
%   >> check_retr
%
% Outputs, all written to results/ :
%   check_retr_n_100_p_10.csv            one row per configuration x retraction
%   check_retr_remainder_n_100_p_10.csv  the three remainder families on the t grid
%   check_retr_n_100_p_10.pdf            the 3 x 3 figure
%   check_retr_console.txt               the Manopt console text
%   check_retr_arrays.mat                the remainder arrays
% The factory also refreshes results/transp_guard_rejections_standalone.csv
% on construction, as it does for every script that builds a manifold.

addpath(pwd);
rng(1);
set(0, 'DefaultFigureVisible', 'off');
script_tic = tic;

%% Problem dimensions and conditioning (same construction as example_trace.m)
n = 100; p = 10;
kappa = 100;

case_list = {'symm', 'skew', 'indef'};
retr_key   = {'cay', 'qgeo', 'pol'};
retr_title = {'Cayley', 'Quasi-geodesic', 'Polar'};

%% Shared SPD cost matrix A (same construction as example_trace.m)
U_seed = sign(rand(n, n) - 0.5);
[U_star, ~, ~] = svds(U_seed, n);
sigma = linspace(1, 1/kappa, n);
A = U_star*diag(sigma)*U_star';

cost_fn  = @(X) trace(X'*A*X);
egrad_fn = @(X) 2*A*X;
ehess_fn = @(X, Xdot) 2*A*Xdot;

output_dir  = 'results';
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

console_path = fullfile(output_dir, 'check_retr_console.txt');
fid_console = fopen(console_path, 'w');
assert(fid_console > 0, 'Could not open console log file for writing.');

nregimes = numel(case_list);
nretr    = numel(retr_key);

t_grid = logspace(-8, 0, 51);
nt     = numel(t_grid);

% Remainder arrays, indexed (t, regime, retraction).
E_X0   = zeros(nt, nregimes, nretr);   % gradient remainder at X0, first-order R
EH_X0  = zeros(nt, nregimes, nretr);   % Hessian remainder at X0, first-order R
EH2_X0 = zeros(nt, nregimes, nretr);   % Hessian remainder at X0, corrected Rtilde

grad_slope_X0  = nan(nregimes, nretr);
hess_slope_X0  = nan(nregimes, nretr);
hess2_slope_X0 = nan(nregimes, nretr);

feasi_X0   = zeros(nregimes, 1);
gamma_norm = zeros(nregimes, 1);       % |Gamma^beta(Z0,Z0)|_F
alpha_norm = zeros(nregimes, nretr);   % |alpha_R(Z0)|_F
feasi_R    = zeros(nregimes, nretr);   % feasibility of R at t = 1
feasi_R2   = zeros(nregimes, nretr);   % feasibility of Rtilde at t = 1

% Reference feasibility of X0 in each regime, recorded from the run that
% produced the published figure. The assertion below, together with the M.hash
% values written to the console log, certifies that a rerun starts from the
% same point.
reference_feasi_X0 = [4.618e-14, 1.470e-13, 5.305e-14];

for ic = 1:nregimes
    regime = case_list{ic};

    %% --- Build Q, P for this regime (same construction as example_trace.m) ---
    switch regime
        case 'symm'
            symm_fn = @(D) sqrtm(D*D') + 1e-6*eye(size(D));
            Q = symm_fn(randn(n)); P = symm_fn(randn(p));
        case 'skew'
            skew_fn = @(D) (D - D')/2;
            Q = skew_fn(randn(n)); P = skew_fn(randn(p));
        case 'indef'
            Uq = orth(randn(n));
            npos = round(0.6*n); nneg = n - npos;
            dQ = [linspace(1, 10, npos), -linspace(1, 10, nneg)];
            Q = Uq*diag(dQ)*Uq'; Q = (Q + Q')/2;
            P = diag([ones(1, p/2), -ones(1, p/2)]);
    end

    %% --- Manifold, problem struct, X0 ---
    M = generalfactory(Q, P);   % beta omitted -> defaults to 1
    M.feasi = @(X) norm(X'*Q*X - P, 'fro');
    M.retr  = M.retr_pol;
    M.transp = M.transp_pol;

    X0 = M.rand();

    problem = struct();
    problem.cost  = cost_fn;
    problem.egrad = egrad_fn;
    problem.ehess = ehess_fn;
    problem.M     = M;

    % One check-manifold per retraction, and one per corrected retraction:
    % M.exp and M.retr2 are removed so that checkdiff.m / checkhessian.m and
    % the local sweeps all step along the retraction named in .retr.
    Mbase = rmfield(M, intersect({'exp', 'retr2'}, fieldnames(M)));
    retr_fun  = {M.retr_cayley, M.retr_qgeo, M.retr_pol};
    retr2_fun = {M.retr2_cay,   M.retr2_qgeo, M.retr2_pol};
    problem_chk  = cell(1, nretr);
    problem_chk2 = cell(1, nretr);
    for ir = 1:nretr
        problem_chk{ir}        = problem;
        problem_chk{ir}.M      = Mbase;
        problem_chk{ir}.M.retr = retr_fun{ir};
        problem_chk2{ir}        = problem;
        problem_chk2{ir}.M      = Mbase;
        problem_chk2{ir}.M.retr = retr2_fun{ir};
    end
    ipol = find(strcmp(retr_key, 'pol'), 1);

    feasi_X0(ic) = M.feasi(X0);

    fprintf(fid_console, '\n==================== REGIME: %s ====================\n', regime);
    fprintf(fid_console, 'n=%d p=%d kappa=%g beta=1\n', n, p, kappa);
    fprintf(fid_console, 'feasi(X0) = %.4e   (reference feasi_X0 = %.4e)\n', ...
        feasi_X0(ic), reference_feasi_X0(ic));
    fprintf(fid_console, 'hash(X0) = %s\n', M.hash(X0));
    assert(abs(feasi_X0(ic) - reference_feasi_X0(ic)) <= 1e-3*reference_feasi_X0(ic), ...
        'check_retr:X0', ['feasi(X0) = %.6e for regime %s does not match the ', ...
        'reference %.6e: the random stream is not the expected one.'], ...
        feasi_X0(ic), regime, reference_feasi_X0(ic));

    Z0 = problem.M.randvec(X0);
    fprintf(fid_console, 'hash(Z0) = %s   |Z0|_F = %.6e\n', M.hash(Z0), norm(Z0, 'fro'));

    %% --- The correction at the seeded direction ---
    gamma_norm(ic) = norm(M.christoffel(X0, Z0, Z0), 'fro');
    for ir = 1:nretr
        alpha_norm(ic, ir) = norm(M.retr2_alpha(X0, Z0, retr_key{ir}), 'fro');
    end
    fprintf(fid_console, ...
        '|Gamma(Z0,Z0)|_F = %.6e   |alpha|_F: cay %.6e  qgeo %.6e  pol %.6e\n', ...
        gamma_norm(ic), alpha_norm(ic, 1), alpha_norm(ic, 2), alpha_norm(ic, 3));

    %% --- checkgradient / checkhessian at X0, first-order polar retraction ---
    %  These two calls also consume the shared random stream, which the
    %  per-regime stream advance below accounts for.
    out_g0 = evalc('checkgradient(problem_chk{ipol}, X0, Z0);');
    fprintf(fid_console, '\n---- checkgradient at X0 (polar retraction) ----\n%s\n', out_g0);

    out_h0 = evalc('checkhessian(problem_chk{ipol}, X0, Z0);');
    fprintf(fid_console, '\n---- checkhessian at X0 (polar retraction) ----\n%s\n', out_h0);
    close all;

    %% --- checkhessian at X0 along the CORRECTED polar retraction ---
    %  Stream-neutral: the state is saved and restored around it, so the
    %  draws checkhessian makes here do not shift X0 in the next regime.
    rng_state = rng;
    out_h2 = evalc('checkhessian(problem_chk2{ipol}, X0, Z0);');
    fprintf(fid_console, ...
        '\n---- checkhessian at X0 (corrected polar retraction) ----\n%s\n', out_h2);
    close all;
    rng(rng_state);

    %% --- Remainder sweeps at X0, one per retraction (no randomness drawn) ---
    for ir = 1:nretr
        [t_E, Ev] = local_compute_E(problem_chk{ir}, X0, Z0);
        assert(isequal(size(t_E), size(t_grid)) && max(abs(t_E - t_grid)) == 0, ...
            'check_retr:grid', 'Gradient-remainder t grid differs from t_grid.');
        E_X0(:, ic, ir) = Ev(:);
        grad_slope_X0(ic, ir) = local_slope(t_grid, Ev);

        [t_EH, EHv] = local_compute_EH(problem_chk{ir}, X0, Z0);
        assert(isequal(size(t_EH), size(t_grid)) && max(abs(t_EH - t_grid)) == 0, ...
            'check_retr:grid', 'Hessian-remainder t grid differs from t_grid.');
        EH_X0(:, ic, ir) = EHv(:);
        hess_slope_X0(ic, ir) = local_slope(t_grid, EHv);

        [~, EH2v] = local_compute_EH(problem_chk2{ir}, X0, Z0);
        EH2_X0(:, ic, ir) = EH2v(:);
        hess2_slope_X0(ic, ir) = local_slope(t_grid, EH2v);

        Y1 = retr_fun{ir}(X0, Z0, 1);
        Y2 = retr2_fun{ir}(X0, Z0, 1);
        feasi_R(ic, ir)  = M.feasi(Y1);
        feasi_R2(ic, ir) = M.feasi(Y2);
    end

    % The polar column must reproduce what Manopt itself printed, for all
    % three rows of the figure.
    local_assert_slope(grad_slope_X0(ic, ipol), out_g0, ...
        'should be 2\.\s*It appears to be:\s*<strong>([^<]+)</strong>', ...
        sprintf('%s grad slope at X0 under R', regime));
    local_assert_slope(hess_slope_X0(ic, ipol), out_h0, ...
        'should be 3\.\s*It appears to be:\s*<strong>([^<]+)</strong>', ...
        sprintf('%s hess slope at X0 under R', regime));
    local_assert_slope(hess2_slope_X0(ic, ipol), out_h2, ...
        'should be 3\.\s*It appears to be:\s*<strong>([^<]+)</strong>', ...
        sprintf('%s hess slope at X0 under Rtilde', regime));

    %% --- Stream advance, so the next regime's X0 is the reference one ---
    Zd = problem.M.randvec(X0);
    out_hd = evalc('checkhessian(problem_chk{ipol}, X0, Zd);');
    fprintf(fid_console, ...
        '\n---- checkhessian at X0, second direction (polar retraction) ----\n%s\n', out_hd);
    out_gd = evalc('checkgradient(problem_chk{ipol}, X0, Zd);');
    fprintf(fid_console, ...
        '\n---- checkgradient at X0, second direction (polar retraction) ----\n%s\n', out_gd);
    close all;
end

fclose(fid_console);

%% ===================================================================
%  Non-finite remainder samples. The corrected retraction has a smaller
%  domain than the retraction it corrects, because the corrected tangent
%  vector Z - alpha(Z)/2 is longer than Z. Where the corrected polar map
%  leaves its domain, retr_pol returns an infeasible point by design and
%  the remainder is not a number. Those samples are recorded as NaN, left
%  out of the plot, and counted in the summary CSV.
%% ===================================================================
nonfinite_EH2 = zeros(nregimes, nretr);
for ic = 1:nregimes
    for ir = 1:nretr
        bad = ~isfinite(EH2_X0(:, ic, ir));
        nonfinite_EH2(ic, ir) = nnz(bad);
        if any(bad)
            fprintf('NOTE: %s / %s : %d non-finite E_H sample(s) under Rtilde at t = %s\n', ...
                case_list{ic}, retr_key{ir}, nnz(bad), ...
                strtrim(sprintf('%.3e ', t_grid(bad))));
            EH2_X0(bad, ic, ir) = NaN;
        end
    end
end

%% ===================================================================
%  CSV output
%% ===================================================================
csv_path = fullfile(output_dir, sprintf('check_retr_n_%d_p_%d.csv', n, p));
fid_csv = fopen(csv_path, 'w');
assert(fid_csv > 0, 'Could not open summary CSV for writing.');
fprintf(fid_csv, ['regime,retraction,grad_slope_X0,hess_slope_X0_retr,', ...
    'hess_slope_X0_retr2,feasi_X0,feasi_retr_t1,feasi_retr2_t1,', ...
    'gamma_norm,alpha_norm,nzero_E_X0,nzero_EH_X0,nzero_EH2_X0,nonfinite_EH2_X0\n']);
for ic = 1:nregimes
    for ir = 1:nretr
        fprintf(fid_csv, '%s,%s,%.6f,%.6f,%.6f,%.3e,%.3e,%.3e,%.6e,%.6e,%d,%d,%d,%d\n', ...
            case_list{ic}, retr_key{ir}, ...
            grad_slope_X0(ic, ir), hess_slope_X0(ic, ir), hess2_slope_X0(ic, ir), ...
            feasi_X0(ic), feasi_R(ic, ir), feasi_R2(ic, ir), ...
            gamma_norm(ic), alpha_norm(ic, ir), ...
            nnz(E_X0(:, ic, ir) == 0), nnz(EH_X0(:, ic, ir) == 0), ...
            nnz(EH2_X0(:, ic, ir) == 0), nonfinite_EH2(ic, ir));
    end
end
fclose(fid_csv);

% Raw remainders: one t column, then the nine E blocks, the nine E_H blocks
% under the first-order retractions and the nine E_H blocks under the
% corrected ones, regime-major with the retraction index running fastest.
% The regime tag "spd" for the symm regime follows check_remainder_n_100_p_10.csv.
reg_tag = {'spd', 'skew', 'indef'};
raw_path = fullfile(output_dir, sprintf('check_retr_remainder_n_%d_p_%d.csv', n, p));
fid_raw = fopen(raw_path, 'w');
assert(fid_raw > 0, 'Could not open raw remainder CSV for writing.');
hdr = 't';
qtag = {'E', 'EH', 'EH2'};
for iq = 1:3
    for ic = 1:nregimes
        for ir = 1:nretr
            hdr = [hdr, ',', qtag{iq}, '_', reg_tag{ic}, '_', retr_key{ir}]; %#ok<AGROW>
        end
    end
end
fprintf(fid_raw, '%s\n', hdr);
flat = @(Z) reshape(permute(Z, [1 3 2]), nt, []);
raw_block = cat(2, flat(E_X0), flat(EH_X0), flat(EH2_X0));
for kk = 1:nt
    fprintf(fid_raw, '%.10e', t_grid(kk));
    fprintf(fid_raw, ',%.10e', raw_block(kk, :));
    fprintf(fid_raw, '\n');
end
fclose(fid_raw);

save(fullfile(output_dir, 'check_retr_arrays.mat'), 't_grid', 'E_X0', 'EH_X0', ...
    'EH2_X0', 'grad_slope_X0', 'hess_slope_X0', 'hess2_slope_X0', ...
    'feasi_X0', 'feasi_R', 'feasi_R2', 'gamma_norm', 'alpha_norm', ...
    'nonfinite_EH2', 'case_list', 'retr_key', 'retr_title');

%% ===================================================================
%  Figure: 3 x 3 grid. Row 1 is the gradient remainder E(t) at X0 under
%  the first-order retraction, row 2 the Hessian remainder E_H(t) at X0
%  under the same retraction, row 3 the Hessian remainder at X0 under its
%  second-order correction. The columns are the three retractions, in the
%  order the paper constructs them. Style constants match example_trace.m.
%% ===================================================================
FONT_NAME = 'Times New Roman';
try
    if ~any(strcmpi(listfonts, FONT_NAME)); FONT_NAME = 'Helvetica'; end
catch
    FONT_NAME = 'Helvetica';
end
FS_TICK = 10; FS_LABEL = 12; FS_TITLE = 13; FS_LEGEND = 12; FS_REF = 9;
LINE_WIDTH = 1.4; MARKER_SIZE = 5; MARKER_NUM = 6;
REF_COL = [0.45 0.45 0.45];

REG_COL = [0.00 0.45 0.74;    % blue      -- symmetric positive-definite
           0.85 0.33 0.10;    % vermilion -- skew-symmetric
           0.47 0.67 0.19];   % green     -- symmetric indefinite
REG_MK  = {'o', 's', '^'};
REG_LEG = {'Symmetric positive-definite', 'Skew-symmetric', 'Symmetric indefinite'};

XLIM = [1e-8, 1e0]; XTICK = 10.^(-8:2:0);
XTICKLAB = {'10^{-8}', '', '10^{-4}', '', '10^{0}'};

nrow = 3;
row_data  = {E_X0, EH_X0, EH2_X0};
row_slope = {grad_slope_X0, hess_slope_X0, hess2_slope_X0};
row_ylab  = {'$E(t)$', '$E_H(t)$', '$E_H(t)$'};
row_ref   = {2, 2, 3};
% Indexed by the slope: REF_GAP(s) is the number of decades by which the
% slope-s guide is placed below the closest approach of any curve, LAB_T(s)
% the abscissa at which the guide is named.
REF_GAP = [NaN, 1.0, 1.0, 1.0];
LAB_T   = [NaN, 1e-3, 3e-2, 3e-2];
% Every row of this figure is taken at X0, so every row bottoms out on the
% rounding floor of its own remainder at small t. FLOOR_MULT sets how far
% above that floor a sample must sit to count as being on the power law and
% so to constrain the guide; FLOOR_SHOW truncates the guide on the left at
% the height where it would meet the floor, so no guide is drawn through it.
FLOOR_MULT = 30; FLOOR_SHOW = 3;

row_ylim = cell(1, nrow); row_ytick = cell(1, nrow);
for irow = 1:nrow
    Y = row_data{irow};
    Yp = Y(isfinite(Y) & Y > 0);
    lo_exp = 2*floor((floor(log10(min(Yp))) - 1)/2);
    hi_exp = 2*ceil((ceil(log10(max(Yp))) + 1)/2);
    row_ylim{irow}  = [10^lo_exp, 10^hi_exp];
    step = 2*ceil((hi_exp - lo_exp)/10);
    row_ytick{irow} = 10.^(lo_exp:step:hi_exp);
end

fig = figure('Color', 'white', 'Units', 'inches', 'Position', [0 0 6.6 5.3]);
tiledlayout(fig, nrow, nretr, 'TileSpacing', 'compact', 'Padding', 'compact');

hleg = gobjects(nregimes, 1);
midx = unique(round(linspace(1, nt, MARKER_NUM)));
axall = gobjects(nrow, nretr);
ref_spec = cell(nrow, nretr);

for irow = 1:nrow
    for ir = 1:nretr
        ax = nexttile; hold(ax, 'on'); box(ax, 'on');
        axall(irow, ir) = ax;

        % Exact zeros and non-finite samples have no place on a log axis.
        % The zeros occur only in the rounding-noise floor and are left as
        % gaps rather than clamped to the axis floor; the non-finite ones
        % are the grid points at which a corrected retraction has left its
        % domain, and are also left as gaps.
        Y = squeeze(row_data{irow}(:, :, ir));
        Y(~isfinite(Y) | Y <= 0) = NaN;

        for ic = 1:nregimes
            h = plot(ax, t_grid, Y(:, ic)', '-', ...
                'Color', REG_COL(ic, :), 'LineWidth', LINE_WIDTH, ...
                'Marker', REG_MK{ic}, 'MarkerSize', MARKER_SIZE, ...
                'MarkerIndices', midx);
            if irow == 1 && ir == 1, hleg(ic) = h; end
        end

        s = row_ref{irow};

        % Rounding-noise level of this panel: the median of each regime's
        % eleven leftmost samples, the largest of the three taken so that
        % the guide clears every regime's floor.
        floor_reg = zeros(1, nregimes);
        for ic = 1:nregimes
            col = Y(1:11, ic); col = col(isfinite(col) & col > 0);
            if isempty(col); col = NaN; end
            floor_reg(ic) = median(col);
        end
        floor_lvl = max(floor_reg);

        ratio = Y ./ (t_grid(:).^s);
        ratio(Y <= FLOOR_MULT*floor_lvl) = NaN;
        cval = 10^(-REF_GAP(s))*min(ratio(:), [], 'omitnan');
        tlo  = max(XLIM(1), (FLOOR_SHOW*floor_lvl/cval)^(1/s));
        tt   = [tlo, XLIM(2)];
        plot(ax, tt, cval*tt.^s, '--', 'Color', REF_COL, ...
            'LineWidth', 1.0, 'HandleVisibility', 'off');
        ref_spec{irow, ir} = [s, cval, tlo];

        set(ax, 'XScale', 'log', 'YScale', 'log', ...
            'FontName', FONT_NAME, 'FontSize', FS_TICK, ...
            'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.12, ...
            'XMinorGrid', 'off', 'YMinorGrid', 'off', ...
            'XMinorTick', 'off', 'YMinorTick', 'off');
        set(ax, 'XLim', XLIM, 'XTick', XTICK, ...
            'YLim', row_ylim{irow}, 'YTick', row_ytick{irow});

        if irow == 1
            title(ax, retr_title{ir}, 'FontName', FONT_NAME, 'FontSize', FS_TITLE);
        end
        if irow < nrow
            set(ax, 'XTickLabel', []);
        else
            set(ax, 'XTickLabel', XTICKLAB);
            xlabel(ax, '$t$', 'Interpreter', 'latex', 'FontSize', FS_LABEL);
        end
        if ir == 1
            ylabel(ax, row_ylab{irow}, 'Interpreter', 'latex', 'FontSize', FS_LABEL);
        else
            set(ax, 'YTickLabel', []);
        end
    end
end

lgd = legend(hleg, REG_LEG, 'Orientation', 'horizontal', ...
    'FontName', FONT_NAME, 'FontSize', FS_LEGEND);
try
    lgd.Layout.Tile = 'south';
catch
    set(lgd, 'Location', 'southoutside');
end

% Labels on the guides. The on-screen angle of a power law of slope s is set
% by the panel's aspect in inches per decade, so it is measured from the
% drawn axes rather than assumed. Each label hangs just below its own line.
drawnow;
fig_sz = get(fig, 'Position');
dec_x  = log10(XLIM(2)/XLIM(1));
for irow = 1:nrow
    dec_y = log10(row_ylim{irow}(2)/row_ylim{irow}(1));
    for ir = 1:nretr
        ax  = axall(irow, ir);
        pos = get(ax, 'Position');
        w_in = pos(3)*fig_sz(3); h_in = pos(4)*fig_sz(4);
        spec = ref_spec{irow, ir};
        s = spec(1); c = spec(2); tlo = spec(3);
        ang = atan2d(s*h_in/dec_y, w_in/dec_x);
        tlab = max(LAB_T(s), 2*tlo);
        while c*tlab^s < row_ylim{irow}(1)*10 && tlab < 0.3
            tlab = tlab*10;
        end
        text(ax, tlab, c*tlab^s, sprintf('slope %d', s), ...
            'FontName', FONT_NAME, 'FontSize', FS_REF, 'Color', REF_COL, ...
            'Rotation', ang, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top');
    end
end

fig_path = fullfile(output_dir, sprintf('check_retr_n_%d_p_%d.pdf', n, p));
exportgraphics(fig, fig_path, 'ContentType', 'vector', 'BackgroundColor', 'white');
close all;

%% --- fitted slopes, echoed to stdout for the run log
fprintf('\nFitted slopes (Manopt window fit, t = logspace(-8,0,51)), all at X0\n');
fprintf('%-7s %-6s %16s %16s %16s\n', 'regime', 'retr', ...
    'E under R', 'E_H under R', 'E_H under Rtilde');
for ic = 1:nregimes
    for ir = 1:nretr
        fprintf('%-7s %-6s %16.6f %16.6f %16.6f\n', case_list{ic}, retr_key{ir}, ...
            grad_slope_X0(ic, ir), hess_slope_X0(ic, ir), hess2_slope_X0(ic, ir));
    end
end
fprintf('\n%-7s %12s %14s %14s %14s %14s\n', 'regime', 'feasi_X0', '|Gamma|', ...
    '|alpha_cay|', '|alpha_qgeo|', '|alpha_pol|');
for ic = 1:nregimes
    fprintf('%-7s %12.3e %14.6e %14.6e %14.6e %14.6e\n', case_list{ic}, ...
        feasi_X0(ic), gamma_norm(ic), alpha_norm(ic, 1), alpha_norm(ic, 2), alpha_norm(ic, 3));
end
fprintf('\nFeasibility at t = 1 (R / Rtilde):\n');
for ic = 1:nregimes
    for ir = 1:nretr
        fprintf('  %-6s %-5s : %.3e / %.3e\n', case_list{ic}, retr_key{ir}, ...
            feasi_R(ic, ir), feasi_R2(ic, ir));
    end
end
fprintf('\nExact-zero remainder samples (count / t values):\n');
qname = {'E@X0 (R)', 'EH@X0 (R)', 'EH@X0 (Rtilde)'};
qdata = {E_X0, EH_X0, EH2_X0};
for iq = 1:3
    for ic = 1:nregimes
        for ir = 1:nretr
            idx = find(qdata{iq}(:, ic, ir) == 0);
            if ~isempty(idx)
                fprintf('  %-14s %-6s %-6s : %d zeros at t = %s\n', qname{iq}, ...
                    case_list{ic}, retr_key{ir}, numel(idx), ...
                    strtrim(sprintf('%.2e ', t_grid(idx))));
            end
        end
    end
end

fprintf('\nWrote %s\n', csv_path);
fprintf('Wrote %s\n', raw_path);
fprintf('Wrote %s\n', fig_path);
fprintf('Elapsed: %.1f s\n', toc(script_tic));
disp('check_retr done.');

%% -------------------- local functions --------------------
function s = local_slope(t, err)
% The slope Manopt's checkdiff.m / checkhessian.m would report for the
% remainder curve (t, err): the slope of the best-fitting line over any
% sliding window of window_len+1 = 11 consecutive grid points, in log-log
% coordinates. When the model is exact to numerical precision Manopt
% declares the slope uninformative and prints no number; NaN is recorded
% here for that case.
    if ~all(err < 1e-12)
        [~, poly] = identify_linear_piece(log10(t), log10(err), 10);
        s = poly(1);
    else
        s = NaN;
    end
end

function local_assert_slope(mine, text, pattern, label)
% Cross-check one locally fitted slope against the number Manopt printed
% for the same curve, so that "same fitting method as Manopt" is a verified
% statement rather than an assumption. Manopt prints with %g, i.e. six
% significant digits, which sets the tolerance.
    tok = regexp(text, pattern, 'tokens', 'once');
    if isempty(tok)
        error('check_retr:parse', ...
            'Expected slope pattern not found in Manopt output for %s.', label);
    end
    theirs = str2double(tok{1});
    if ~(abs(mine - theirs) <= 1e-5*max(1, abs(theirs)))
        error('check_retr:slopemismatch', ...
            'Local slope %.8g disagrees with Manopt''s %.8g for %s.', mine, theirs, label);
    end
end

function [t, E] = local_compute_E(problem, x, d)
% Recomputes the (t, E(t)) array that Manopt's checkdiff.m computes
% internally but does not return to its caller. checkgradient.m calls
% checkdiff.m with force_gradient = true, i.e. with df0 = <grad f(x), d>,
% which is what is used here. Same t = logspace(-8, 0, 51) grid, same
% linear Taylor model, and problem.M.retr as the stepper.
    storedb = StoreDB();
    xkey  = storedb.getNewKey();
    f0    = getCost(problem, x, storedb, xkey);
    gradx = getGradient(problem, x, storedb, xkey);
    df0   = problem.M.inner(x, gradx, d);

    t = logspace(-8, 0, 51);
    value = zeros(size(t));
    for i = 1:length(t)
        y = problem.M.retr(x, d, t(i));
        ykey = storedb.getNewKey();
        value(i) = getCost(problem, y, storedb, ykey);
        storedb.remove(ykey);
    end
    model = polyval([df0, f0], t);
    E = abs(model - value);
end

function [t, EH] = local_compute_EH(problem, x, d)
% Recomputes the (t, E_H(t)) array that Manopt's checkhessian.m computes
% internally but does not return to its caller. Same grid, same quadratic
% Taylor model, and problem.M.retr as the stepper.
    storedb = StoreDB();
    xkey = storedb.getNewKey();
    f0     = getCost(problem, x, storedb, xkey);
    gradx  = getGradient(problem, x, storedb, xkey);
    df0    = problem.M.inner(x, d, gradx);
    hessxd = getHessian(problem, x, d, storedb, xkey);
    d2f0   = problem.M.inner(x, d, hessxd);

    t = logspace(-8, 0, 51);
    value = zeros(size(t));
    for i = 1:length(t)
        y = problem.M.retr(x, d, t(i));
        ykey = storedb.getNewKey();
        value(i) = getCost(problem, y, storedb, ykey);
        storedb.remove(ykey);
    end
    model = polyval([.5*d2f0, df0, f0], t);
    EH = abs(model - value);
end
