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
% The constraint spectra are designed by solving an affine linear program with
% MATLAB's linprog and a secondary minimum-norm quadratic program with quadprog.
% CVX is not required. The baseline manifold is symplecticfactory.m and the
% skew-symmetric eigendecompositions use YoulaDecomposition.m.
%
% Reproduces the skew-symmetric half of the numerical experiments of the paper
% "Preconditioning for Quadratic Matrix Optimization via Constraint
% Reshaping" (arXiv:2611.26078).
%
% Seed rng(0), set once at the top. Dimensions n = 100, p = 20. The Hessian
% spectra dominate the cost of the run.
%
% Requires Manopt 8.0, the Optimization Toolbox (linprog, quadprog), and the
% Control System Toolbox (lyap, through the factories).
%
% The TIMING-BEGIN and TIMING-END comments delimit the blocks used by the
% paper's timing protocol; they are comments only.
%
% Run from this directory:
%   >> example_skew
%
% Outputs, all written to results/ , with <problem> ranging over eigenvalue,
% brockett, tridiagonal and quadratic:
%   skew_<problem>_n_100_p_20.pdf             convergence figure
%   skew_<problem>_n_100_p_20.csv             matching table
%   skew_<problem>_n_100_p_20_timing.csv      timing breakdown
%   skew_<problem>_n_100_p_20_validation.mat  full run record
%   legend_skew.pdf                           shared legend strip

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
% TIMING-BEGIN
timing_init_clock = tic;
% TIMING-END
W = randn(p,p);
E = expm(Jp*(W'*W + eye(p)/10));
Y0 = [E(1:p/2,:);zeros((n-p)/2,p);E(p/2+1:end,:);zeros((n-p)/2,p)];
% TIMING-BEGIN
timing_initial = toc(timing_init_clock);
% TIMING-END

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
% Plotting constants of the IMANUM v2_15 figure layout, taken from
% review/IMANUM/v2/v2_15-apply/replot_figs.m. At 7 in, placed at \linewidth
% (145 mm) in the paper, a design size s prints at about 0.81 s, so ticks print
% at 8.5 pt and axis labels and the legend at 8.9 pt. The legend is exported once
% per family as a separate strip, results/legend_skew.pdf, shared by the four
% instance figures. The PDFs committed in results/ predate this layout: v2_15
% redrew the published figures from their vector data (GQ-IMANUM/figures/)
% without rerunning this driver, so a rerun regenerates them in the new layout.
FONT_NAME = 'Times New Roman';
FONT_SIZE_LABEL = 11;     % prints at 8.9 pt
FONT_SIZE_TICK = 10.5;    % prints at 8.5 pt
FONT_SIZE_LEGEND = 11;    % prints at 8.9 pt
AXIS_COLOR = [0.15 0.15 0.15];
LINE_WIDTH = 2;
AXIS_LINE_WIDTH = 1.0;
MARKER_SIZE = 4;
MARKER_NUM = 5;
FIG_WIDTH = 7;
FIG_HEIGHT = 1.9;

left = 0.075;   % v2_15 layout (replot_figs.m)
right = 0.012;
bottom = 0.25;  % room for the x labels only, the legend is a separate strip
top = 0.035;
hspace = 0.078;
vspace = 0.01;  % 子图垂直间距
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
% TIMING-BEGIN
    timing_setup = zeros(1, 5);
    timing_method_clock = tic;
% TIMING-END
    sol{1}.M = symplecticfactory(n/2, p/2, 'canonical');
    sol{1}.cost = @(Y) trace(Y'*C*Y*D);
    sol{1}.egrad = @(Y) 2*C*Y*D;
    sol{1}.ehess = @(Y, Ydot) 2*C*Ydot*D;
    sol{1}.Y0 = Y0;
    % feasi = ||X^{(P,Q)}X - I_p||_F = ||P^{-1}(X'QX - P)||_F; here P = Jp, Q = Jn.
    sol{1}.feasi = @(Y) norm(Jp\(Y'*Jn*Y - Jp), 'fro');
    sol{1}.name = "Sp(p,n)";
    sol{1}.condhat = @(x) nan;
% TIMING-BEGIN
    timing_setup(1) = toc(timing_method_clock);
    timing_method_clock = tic;
% TIMING-END

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
    sol{2}.condhat = @(x) GQ_spectrum(c, d, d, c, n, p);
% TIMING-BEGIN
    timing_setup(2) = toc(timing_method_clock);
    timing_method_clock = tic;
% TIMING-END

    % Optimized design with native sparse LP and quadratic secondary program
    rootC = sqrtm(C);
    rootD = sqrtm(D);
    frakC = rootC\Jn/rootC;
    frakD = rootD*Jp*rootD;
    [UC, dEC] = YoulaDecomposition(frakC);
    [UD, dED] = YoulaDecomposition(frakD);
    EC = diag(dEC(1:2:n, (1:2:n)+1));
    ED = diag(dED(1:2:p, (1:2:p)+1));

    % Same affine mode model and secondary objective with native solvers.
    design_rng_before = rng;
    [design, design_state] = native_design_fast('skew', EC, ED);
    design_rng_after = rng;
    assert(isequal(design_rng_before, design_rng_after), 'Native design changed advancing RNG stream');
    design_x = design.x;
    design_y = design.y;
    x = design_x;
    y = design_y;
    opt_lp = design.lp_value;
    slack = design.slack;

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
    sol{3}.condhat = @(x) GQ_spectrum(c, d, EP, EQ, n, p);
% TIMING-BEGIN
    timing_setup(3) = toc(timing_method_clock);
% TIMING-END

    fstar = 2*sum( 1./c(1:p/2) .* d(1:p/2));
    disp(fstar)

% TIMING-BEGIN
    timing_options_clock = tic;
% TIMING-END
    opt.verbosity = 0;
    opt.maxiter = 1e4;
    opt.tolgradnorm = 1e-6;   % set explicitly (equals the Manopt default) for reproducibility
% TIMING-BEGIN
    timing_options = toc(timing_options_clock);
% TIMING-END
    lens = numel(sol);
    out_infos = cell(lens, 1);
    out_lambdas = cell(lens, 1);
    timing_raw_spectra = cell(lens, 1);
    timing_zero_info = cell(lens, 1);
    % out_conds = nan(lens, 1);
    condstr = cell(lens, 1);
    % Semantic method labels for the CSV row names and the figure legend, matching
    % the published artefacts (SET-08: the shipped script had regressed these to
    % 'method 1..3'; the committed CSVs and PDFs use these names). Order = sol{1..3}.
    method_labels = {'Sp', 'PC-bal', 'PC-opt'};
    assert(numel(method_labels) == lens, 'method_labels count must equal lens');
% TIMING-BEGIN
    timing_shared = zeros(1, lens);
    timing_preprocessing = timing_initial + timing_shared + timing_setup(1:lens) + timing_options;
    design_diagnostics = native_design_diagnostics(design_state);
    fprintf('DESIGN %s LP %.17g QP %.17g slack %.17g attempts %d primal %.3g\n', ...
        filename, design.lp_value, design.qp_value, design.slack, ...
        design.stage2_attempts, design_diagnostics.qp_primal_violation);

    timing_breakdown = zeros(lens, 10);
    timing_points = cell(lens, 1);
    timing_histories = cell(lens, 1);
    timing_rng_states = cell(lens, 1);
    timing_shifted_histories = cell(lens, 1);
    timing_resolved_options = cell(lens, 1);
    timing_stopping = cell(lens, 1);
    timing_design_bases = struct('UQ', UQ, 'UP', UP);
    timing_initial_mappings = struct('PC_bal_Q',tildeQ,'PC_bal_P',tildeP,'PC_opt_Q',breveQ,'PC_opt_P',breveP);
    timing_initial_points = cellfun(@(s) s.Y0, sol, 'UniformOutput', false);
% TIMING-END
    T_vals = cell(lens, 1);
    for is = 1:lens
        solver = sol{is};
        % [x, cost, info, ~] = trustregions(solver, solver.Y0, opt);
% TIMING-BEGIN
        timing_rng_states{is} = rng;
        timing_solver_clock = tic;
% TIMING-END
        [x, cost, info, solver_options] = conjugategradient(solver, solver.Y0, opt);
% TIMING-BEGIN
        timing_solver_wall = toc(timing_solver_clock);
        timing_solver_info = info(end).time;
        timing_solver_overhead = timing_solver_wall - timing_solver_info;
        assert(timing_solver_overhead >= -1e-8, 'Outer solver clock shorter than reported clock');
        timing_total = timing_preprocessing(is) + timing_solver_wall;
        timing_points{is} = x;
        timing_histories{is} = info;
        timing_shifted_histories{is} = timing_preprocessing(is) + timing_solver_overhead + [info.time];
        assert(abs(timing_shifted_histories{is}(end) - timing_total) <= 1e-10 * max(1, timing_total));
        timing_breakdown(is, :) = [timing_initial, timing_shared(is), timing_setup(is), timing_options, timing_preprocessing(is), timing_solver_info, timing_solver_overhead, timing_solver_wall, timing_total, timing_total / info(end).iter];
% TIMING-END
        timing_resolved_options{is} = solver_options;
        timing_stopping{is} = record_solver_termination(solver, x, solver_options, info);
        fprintf('ALGORITHM %s %s iter %d total %.9g reason %s\n', filename, ...
            method_labels{is}, info(end).iter, timing_total, timing_stopping{is}.label);
        [cond, lambda, zinfo, lambda_raw] = getcond(solver, x);
        timing_raw_spectra{is} = lambda_raw;
        timing_zero_info{is} = zinfo;
        fprintf('zero-modes %-8s tol=%.3e kept_min=%.6e rejected_max=%.3e n_rejected=%d\n', ...
            method_labels{is}, zinfo.zero_tol, zinfo.kept_min, zinfo.rejected_max, zinfo.n_rejected);
        out_infos{is} = info;
        out_lambdas{is} = lambda;
        condstr{is} = ['\textsf{' method_labels{is} '}'];
        % condstr{is} = sprintf('method %d ($\\kappa=%.2e$)', is, cond);
        T_vals{is} = [solver.cost(x), info(end).gradnorm, ...
            solver.feasi(x), info(end).iter, timing_total,...
            timing_total/info(end).iter, cond, solver.condhat(x)];
    end

    flush_transp_guard_log(fullfile(output_dir, 'transp_guard_rejections_skew.csv'));

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
    light_theme(fig);
    set(fig, 'PaperUnits','inches','PaperSize',[FIG_WIDTH FIG_HEIGHT],...
        'PaperPosition',[0 0 FIG_WIDTH FIG_HEIGHT]);
    ax = gobjects(3, 1);

    ax(1) = axes('Units','normalized','Position', [posx(1), posy, subplotW, subplotH]);
    hold on; box on; grid off;

    for is = 1:lens
        out_info = out_infos{is};
        marker_idx = round(logspace(0, log10(length([out_info.time])), MARKER_NUM));
        % marker_idx = round(linspace(1, length([out_info.time]), MARKER_NUM));
        semilogy(timing_preprocessing(is) + timing_breakdown(is, 7) + [out_info.time], [out_info.cost]-fstar, 'LineStyle', line_list{is},...
            'Color', color_list{is}, 'Marker', marker_list{is}, ...
            'LineWidth', LINE_WIDTH, 'MarkerSize', MARKER_SIZE, ...
            'MarkerIndices', marker_idx);
        % hold on; box on; grid off;
    end
    set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize',FONT_SIZE_TICK, ...
        'FontName', FONT_NAME, 'LineWidth', AXIS_LINE_WIDTH, ...
        'Color', 'w', 'XColor', AXIS_COLOR, 'YColor', AXIS_COLOR);
    decade_ticks(gca);
    xlabel('Total time [s]', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL, 'Color',AXIS_COLOR);
    ylabel('Cost value gap', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL, 'Color',AXIS_COLOR);

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
        'FontName', FONT_NAME, 'LineWidth', AXIS_LINE_WIDTH, ...
        'Color', 'w', 'XColor', AXIS_COLOR, 'YColor', AXIS_COLOR);
    decade_ticks(gca);
    xlabel('Iteration #', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL, 'Color',AXIS_COLOR);
    ylabel('Gradient norm', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL, 'Color',AXIS_COLOR);

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
        'FontName', FONT_NAME, 'LineWidth', AXIS_LINE_WIDTH, ...
        'Color', 'w', 'XColor', AXIS_COLOR, 'YColor', AXIS_COLOR);
    decade_ticks(gca);
    xlabel('Eigenvalue number (sorted)', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL, 'Color',AXIS_COLOR);
    ylabel('Hessian eigenvalues', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL, 'Color',AXIS_COLOR);

    filename_csv = fullfile(output_dir, sprintf('%s.csv', filename));
    filename_pdf = fullfile(output_dir, sprintf('%s.pdf', filename));

    writetable(T, filename_csv, 'WriteRowNames', true);
% TIMING-BEGIN
    timing_columns = {'initial_point', 'shared_spectral', 'method_setup', 'options', 'preprocessing', 'solver_info', 'solver_unlogged_overhead', 'solver_wall', 'total', 'total_per_iter'};
    timing_table = array2table(timing_breakdown, 'VariableNames', timing_columns, 'RowNames', method_labels(:));
    writetable(timing_table, fullfile(output_dir, [filename '_timing.csv']), 'WriteRowNames', true);
    timing_final_rng = rng;
    save(fullfile(output_dir, [filename '_validation.mat']), 'timing_points', 'timing_histories', 'timing_shifted_histories', 'timing_rng_states', 'timing_final_rng', 'timing_table', 'out_lambdas', 'opt', 'C', 'D', 'Y0', 'EP', 'EQ', 'x', 'y', 'slack', 'opt_lp', 'design_x', 'design_y', 'design', 'design_diagnostics', 'design_state', 'timing_resolved_options', 'timing_stopping', 'timing_design_bases', 'timing_initial_points', 'timing_initial_mappings', 'design_rng_before', 'design_rng_after', 'timing_raw_spectra', 'timing_zero_info');
% TIMING-END
    exportgraphics(fig, filename_pdf, 'ContentType', 'vector', ...
        'BackgroundColor', 'none', 'Resolution', 300);
end

%% Shared legend strip for the four instance figures (v2_15 layout)
fig = figure('Color','white','Units','inches', 'Position',[0 0 FIG_WIDTH 0.3]);
light_theme(fig);
set(fig, 'PaperUnits','inches','PaperSize',[FIG_WIDTH 0.3],...
    'PaperPosition',[0 0 FIG_WIDTH 0.3]);
leg_ax = axes('Position', [0 0 1 1], 'Visible', 'off');
hold on;
h = gobjects(lens, 1);
for is = 1:lens
    h(is) = plot(leg_ax, NaN, NaN, 'LineStyle', line_list{is},...
        'Color', color_list{is}, 'Marker', marker_list{is}, ...
        'LineWidth', LINE_WIDTH, 'MarkerSize', MARKER_SIZE);
end
lg = legend(leg_ax, h, condstr, 'Orientation', 'horizontal', 'Box', 'on', ...
    'FontName', FONT_NAME, 'FontSize', FONT_SIZE_LEGEND, 'Interpreter', 'latex', ...
    'Color', 'w', 'TextColor', [0 0 0], 'EdgeColor', AXIS_COLOR);
lg.Units = 'normalized';
lg.Position(1) = 0.5 - lg.Position(3)/2;
lg.Position(2) = 0.5 - lg.Position(4)/2;
exportgraphics(fig, fullfile(output_dir, 'legend_skew.pdf'), 'ContentType', 'vector', ...
    'BackgroundColor', 'none');
close(fig);

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
function [cond, lambda, zinfo, lambda_raw] = getcond(solver, x)
    lambda = hessianspectrum(solver, x);
    lambda_raw = lambda;
    % Retain genuine small positive modes. A fixed 1e-5 cutoff deleted the
    % true 8.93e-6 minimum mode in the skew tridiagonal PC-bal experiment.
    zero_tol = 1e-12 * max([1; abs(lambda(:))]);
    lambda = real(lambda);
    rejected = lambda(lambda <= zero_tol);
    lambda = lambda(lambda > zero_tol);
    lambda = sort(lambda, 'descend');
    cond = lambda(1) / lambda(end);
    % RV4.2-2: archive the classification boundary, so a reader can check that no
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


%%
% ======================================================================
% Plotting helpers of the v2_15 figure layout (replot_figs.m).
% ======================================================================
function decade_ticks(ax)
% Major ticks at powers of ten, at most four labelled decades per axis. An axis
% that spans fewer than two decades keeps MATLAB's automatic ticks.
lims = {get(ax, 'XLim'), get(ax, 'YLim')};
props = {'XTick', 'YTick'};
for a = 1:2
    lim = lims{a};
    d = ceil(log10(lim(1)) - 1e-9):floor(log10(lim(2)) + 1e-9);
    if numel(d) < 2, continue; end
    step = max(1, ceil(numel(d) / 4));
    k = d(mod(d, step) == 0);
    if numel(k) < 2, k = d(1:step:end); end
    set(ax, props{a}, 10.^k);
end
end

function light_theme(fig)
% MATLAB R2025a in -batch mode may default to the dark theme.
try
    fig.Theme = 'light';
catch
end
end
