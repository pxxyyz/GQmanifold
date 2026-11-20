% example_symm.m
%
% Preconditioning experiment, symmetric case. On four quadratic objectives
% (eigenvalue, Brockett, tridiagonal, quadratic) the baseline Stiefel manifold
% St(p,n) is compared with the reshaped constraint Gq(P,Q) under a balanced and
% under an optimally designed choice of the constraint spectra, and with the
% Lagrangian-induced and the left-right preconditioned metrics on St(p,n). Each
% run records the Riemannian gradient norm along the iterate sequence and the
% condition number of the Riemannian Hessian at the converged point.
%
% The constraint spectra are designed by solving an affine linear program with
% MATLAB's linprog and a secondary minimum-norm quadratic program with quadprog.
% CVX is not required.
%
% Reproduces the symmetric half of the numerical experiments of the paper
% "Preconditioning for Ill-Conditioned Quadratic Optimization via Reshaping
% Constraints".
%
% Seed rng(0), set once at the top. Dimensions n = 100, p = 20. The Hessian
% spectra dominate the cost of the run.
%
% Requires Manopt 8.0, the Optimization Toolbox (linprog, quadprog), and the
% Control System Toolbox (lyap, through generalfactory).
%
% The TIMING-BEGIN and TIMING-END comments delimit the blocks used by the
% paper's timing protocol; they are comments only.
%
% Run from this directory:
%   >> example_symm
%
% Outputs, all written to results/ , with <problem> ranging over eigenvalue,
% brockett, tridiagonal and quadratic:
%   symm_<problem>_n_100_p_20.pdf             convergence figure
%   symm_<problem>_n_100_p_20.csv             matching table
%   symm_<problem>_n_100_p_20_timing.csv      timing breakdown
%   symm_<problem>_n_100_p_20_validation.mat  full run record
%   legend_symm.pdf                           shared legend strip

clear; close all; clc; rng(0);

%% Create the manifold
% n = 20; p = 6;
% n = 40; p = 10;
n = 100; p = 20;
In = eye(n); Ip = eye(p);

C = gallery('lehmer', n);
% TIMING-BEGIN
timing_init_clock = tic;
% TIMING-END
Y0 = orth(randn(n,p));
% TIMING-BEGIN
timing_initial = toc(timing_init_clock);
% TIMING-END
% Y0 = qr_unique(randn(n, p));

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
% per family as a separate strip, results/legend_symm.pdf, shared by the four
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

% WP10: these depend only on C, not on ip. Hoisted out of the loop; they
% consume no random numbers, so the rng(0) stream is untouched and every
% published number is bit-identical.
% TIMING-BEGIN
timing_shared_clock = tic;
% TIMING-END
[Uc, Lc] = eig(C);
lc = diag(Lc);                % eigenvalues of C (column vector)
lcmin = min(lc);              % lambda_min(C), used by omega(Y)
% TIMING-BEGIN
timing_eigC = toc(timing_shared_clock);
% TIMING-END
c = sort(1 ./ lc, 'descend');     % eig(C^{-1}) descending, length n
% TIMING-BEGIN
timing_shared_clock = tic;
% TIMING-END
rootC = sqrtm(C);
% TIMING-BEGIN
timing_rootC = toc(timing_shared_clock);
% TIMING-END
% TIMING-BEGIN
timing_shared_clock = tic;
% TIMING-END
tildeQ = inv(rootC);
% TIMING-BEGIN
timing_inverse_rootC = toc(timing_shared_clock);
% TIMING-END

for ip = 1:lenp
    D = prob{ip}.D;
    filename = sprintf('symm_%s_n_%d_p_%d', prob{ip}.name, n, p);
    disp(filename);

    d = sort(eig(D),  'descend');     % eig(D) descending, length p
    %% Create the solver
% TIMING-BEGIN
    timing_setup = zeros(1, 5);
    timing_method_clock = tic;
% TIMING-END
    sol{1}.M = stiefelfactory(n, p);
    sol{1}.cost = @(Y) trace(Y'*C*Y*D);
    sol{1}.egrad = @(Y) 2*C*Y*D;
    sol{1}.ehess = @(Y, Ydot) 2*C*Ydot*D;
    sol{1}.Y0 = Y0;
    % feasi = ||X^{(P,Q)}X - I_p||_F = ||P^{-1}(X'QX - P)||_F; here P = Ip, Q = In.
    sol{1}.feasi = @(Y) norm(Y'*Y - Ip, 'fro');
    sol{1}.name = "St(p,n)";
    sol{1}.condhat = @(x) euclid_spectrum(c, d, n, p);
% TIMING-BEGIN
    timing_setup(1) = toc(timing_method_clock);
    timing_method_clock = tic;
% TIMING-END

    % A = I, B = I     (tildeQ = inv(sqrtm(C)) hoisted above: depends only on C)
    tildeP = sqrtm(D);
    Q = symm(tildeQ'*tildeQ);
    P = symm(tildeP'*tildeP);
    A = tildeQ'*C*tildeQ;
    B = tildeP\D/tildeP';
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

    % Optimized design with native sparse LP and quadratic secondary program     (rootC = sqrtm(C) hoisted above: depends only on C)
    rootD = sqrtm(D);
    frakC = rootC\In/rootC;
    frakD = rootD*Ip*rootD;

    [UC, dEC] = eig(frakC, 'vector');
    [~, ind] = sort(dEC,'descend');
    EC = dEC(ind); UC = UC(:,ind);
    [UD, dED] = eig(frakD, 'vector');
    [~, ind] = sort(dED,'descend');
    ED = dED(ind); UD = UD(:,ind);
    % Same affine mode model and secondary objective with native solvers.
    design_rng_before = rng;
    [design, design_state] = native_design_fast('symm', EC, ED);
    design_rng_after = rng;
    assert(isequal(design_rng_before, design_rng_after), 'Native design changed advancing RNG stream');
    design_x = design.x;
    design_y = design.y;
    x = design_x;
    y = design_y;
    opt_lp = design.lp_value;
    slack = design.slack;

    UQ = orth(rand(n));
    EQ = [ones(p,1); exp(y)];
    Q = symm(UQ*diag(EQ)*UQ');

    UP = orth(rand(p));
    EP = exp(x);
    P = symm(UP*diag(EP)*UP');

    breveQ = rootC\UC*diag(EQ./EC).^.5*UQ';
    breveP = rootD*UD*diag(EP./ED).^.5*UP';

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
    timing_method_clock = tic;
% TIMING-END

    eta   = 0.8;                  % safety factor for omega(Y), in (0,1)
    opGLag = @(Y, Z) 2*(C*Z)*D - 2*omega_of_Y(Y, C, D, lcmin, eta)*Z*symm(Y'*C*Y*D);

    MLag = stiefelfactory(n, p);
    MLag.inner = @(Y, Z1, Z2) Z1(:)' * reshape(opGLag(Y, Z2), [], 1);
    MLag.norm  = @(Y, Z) sqrt(max(0, MLag.inner(Y, Z, Z)));
    MLag.egrad2rgrad = @(Y, eg) egrad2rgrad_Lag(Y, eg, C, D, ...
                                  omega_of_Y(Y, C, D, lcmin, eta), Uc, lc, p);
    MLag.ehess2rhess = @(Y, eg, eh, Z) ehess2rhess_Lag(Y, eg, eh, Z, C, D, ...
                                  omega_of_Y(Y, C, D, lcmin, eta), Uc, lc, p);

    sol{4}.M = MLag;
    sol{4}.cost = @(Y) trace(Y'*C*Y*D);
    sol{4}.egrad = @(Y) 2*C*Y*D;
    sol{4}.ehess = @(Y, Ydot) 2*C*Ydot*D;
    sol{4}.Y0 = Y0;
    sol{4}.feasi = @(Y) norm(Y'*Y - Ip, 'fro');
    sol{4}.name = "St(p,n)";
    % The closed-form Lagrangian estimate depends on omega(Y*) = omega_of_Y
    % evaluated AT THE OPTIMUM, which is only available after the solve.
    % Defer it as a function of the converged point x (resolved in the loop).
    sol{4}.condhat = @(x) lag_spectrum(c, d, n, p, ...
                                       omega_of_Y(x, C, D, lcmin, eta));
    % WP10: hessianspectrum evaluates the Hessian ~2001 times at ONE fixed base
    % point, and egrad2rgrad_Lag rebuilds the same p(p+1)/2-square tangency
    % system every time. Hand getcond a prepared solver whose cached operator
    % reuses that system. Bit-identical by construction (verified: the cached
    % and uncached rgrad differ by exactly 0 in Frobenius norm).
    sol{4}.prepare_hessian = @(s, x) prepare_lag_hessian(s, x, C, D, ...
                                       lcmin, eta, Uc, lc, p);
% TIMING-BEGIN
    timing_setup(4) = toc(timing_method_clock);
    timing_method_clock = tic;
% TIMING-END

    delta = 1e-3;                 % left-right regularisation parameter, > 0
    opGLR = @(Y, Z) (C*Z) * sqrtmL(symm(Y'*C*Y*D), delta, p);

    MLR = stiefelfactory(n, p);
    MLR.inner = @(Y, Z1, Z2) Z1(:)' * reshape(opGLR(Y, Z2), [], 1);
    MLR.norm  = @(Y, Z) sqrt(max(0, MLR.inner(Y, Z, Z)));
    MLR.egrad2rgrad = @(Y, eg) egrad2rgrad_LR(Y, eg, C, D, delta, Uc, lc, p);
    MLR.ehess2rhess = @(Y, eg, eh, Z) ehess2rhess_LR(Y, eg, eh, Z, C, D, delta, Uc, lc, p);
    
    sol{5}.M = MLR;
    sol{5}.cost = @(Y) trace(Y'*C*Y*D);
    sol{5}.egrad = @(Y) 2*C*Y*D;
    sol{5}.ehess = @(Y, Ydot) 2*C*Ydot*D;
    sol{5}.Y0 = Y0;
    sol{5}.feasi = @(Y) norm(Y'*Y - Ip, 'fro');
    sol{5}.name = "St(p,n)";
    sol{5}.condhat = @(x) lr_spectrum(c, d, n, p, delta);
% TIMING-BEGIN
    timing_setup(5) = toc(timing_method_clock);
% TIMING-END

    muD = sort(eig(D), 'descend');
    lcA = sort(lc, 'ascend');
    fstar = sum(lcA(1:p) .* muD);

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
    % 'method 1..5'; the committed CSVs and PDFs use these names). Order = sol{1..5}.
    method_labels = {'St', 'PC-bal', 'PC-opt', 'PM-Lag', 'PM-LR'};
    assert(numel(method_labels) == lens, 'method_labels count must equal lens');
% TIMING-BEGIN
    timing_shared = [0, timing_rootC + timing_inverse_rootC, timing_rootC, timing_eigC, timing_eigC];
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
        % lambda = nan;
        % cond = nan;
        out_infos{is} = info;
        out_lambdas{is} = lambda;
        condstr{is} = ['\textsf{' method_labels{is} '}'];
        % condstr{is} = sprintf('method %d ($\\kappa=%.2e$)', is, cond);
        T_vals{is} = [solver.cost(x), info(end).gradnorm, ...
            solver.feasi(x), info(end).iter, timing_total,...
            timing_total/info(end).iter, cond, solver.condhat(x)];
    end

    flush_transp_guard_log(fullfile(output_dir, 'transp_guard_rejections_symm.csv'));

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
exportgraphics(fig, fullfile(output_dir, 'legend_symm.pdf'), 'ContentType', 'vector', ...
    'BackgroundColor', 'none');
close(fig);


%%
% ======================================================================
% Condition number of the Riemannian Hessian via Manopt's hessianspectrum.
% ======================================================================
function [cond, lambda, zinfo, lambda_raw] = getcond(solver, x)
    % WP10: let a solver install a base-point-specific fast Hessian before the
    % spectrum is taken. The prepared operator must be mathematically identical
    % to the original; it exists only to hoist work that does not depend on the
    % direction being applied.
    if isfield(solver, 'prepare_hessian')
        solver = solver.prepare_hessian(solver, x);
    end
    lambda = hessianspectrum(solver, x);
    lambda_raw = lambda;
    % Scale-aware zero classification, identical to example_skew.m. The former
    % absolute 1e-5 cutoff is scale dependent: it deleted the true 8.93e-6
    % minimum mode in the skew tridiagonal PC-bal experiment.
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
% Closed-form Euclidean Hessian spectrum at the optimum.
%   c = eig(C^{-1}) desc (length n), d = eig(D) desc (length p).
% ======================================================================
function s = euclid_spectrum(c, d, n, p)
    s = [];
    for i = 1:p
        for j = i+1:p
            s(end+1) = (c(i)-c(j))*(d(i)-d(j))/(c(i)*c(j)); %#ok<AGROW>
        end
    end
    for i = 1:p
        for j = p+1:n
            s(end+1) = 2*( d(i)/c(j) - d(i)/c(i) ); %#ok<AGROW>
        end
    end
    s = condfromspec(s(:));
end

% ======================================================================
% Closed-form Gq(P, Q) Hessian spectrum at the optimum.
% ======================================================================
function s = GQ_spectrum(c, d, x, y, n, p, beta)
    if nargin < 7 || isempty(beta), beta = 1; end
    s = [];
    for i = 1:p
        for j = i+1:p
            s(end+1) = (c(i)-c(j))*(d(i)-d(j))/(beta*c(i)*c(j)*x(i)*x(j)); %#ok<AGROW>
        end
    end
    for i = 1:p
        for j = p+1:n
            s(end+1) = 2*d(i)*y(j)*(c(i)-c(j)) /(x(i)*c(i)*c(j)); %#ok<AGROW>
        end
    end
    s = condfromspec(s(:));
end

% ======================================================================
% Closed-form Lagrangian-preconditioned Hessian spectrum at the optimum.
% ======================================================================
function s = lag_spectrum(c, d, n, p, omega)
    s = [];
    for i = 1:p
        for j = i+1:p
            num = (d(i)-d(j))*(c(i)-c(j));
            den = (c(i)*d(i) + c(j)*d(j)) - omega*(c(i)*d(j) + c(j)*d(i));
            s(end+1) = num/den; %#ok<AGROW>
        end
    end
    for i = 1:p
        for j = p+1:n
            s(end+1) = (c(i)-c(j))/(c(i)-omega*c(j)); %#ok<AGROW>
        end
    end
    s = condfromspec(s(:));
end

% ======================================================================
% Closed-form left-and-right preconditioned Hessian spectrum at the optimum.
%   s_k = sqrt( (d_k/c_k)^2 + delta ).
% ======================================================================
function s = lr_spectrum(c, d, n, p, delta)
    sk = sqrt((d(:)./c(1:p)).^2 + delta);   % length p
    s = [];
    for i = 1:p
        for j = i+1:p
            num = 2*(c(i)-c(j))*(d(i)-d(j));
            den = c(i)*sk(i) + c(j)*sk(j);
            s(end+1) = num/den; %#ok<AGROW>
        end
    end
    for i = 1:p
        for j = p+1:n
            s(end+1) = 2*d(i)*(c(i)-c(j))/(c(i)*sk(i)); %#ok<AGROW>
        end
    end
    s = condfromspec(s(:));
end

% ======================================================================
% Condition number from a closed-form spectrum (drop structural zeros).
% ======================================================================
function k = condfromspec(s)
    s = s(abs(s) > 1e-12);
    k = max(s)/min(s);
end

% ======================================================================
% Symmetric part
% ======================================================================
function S = symm(A)
    S = (A + A')/2;
end

% ======================================================================
% State-dependent regularisation for the Lagrangian-induced metric.
%
%   omega(Y) = eta * omega_crit(Y),   eta in (0,1) a safety factor,
%   with the EXACT admissibility ceiling
%       omega_crit(Y) = lambda_min(C) / lambda_max(L, D),
%   where L = sym(Y'*C*Y*D) and lambda_max(L, D) is the largest generalized
%   eigenvalue of the pencil (L, D), i.e. the largest mu solving L v = mu D v.
%
%   Rationale: H_Lag(Y)[Z] = 2 C Z D - 2 omega Z L is positive definite iff
%   every C-eigenblock  2 lc(a) D - 2 omega L  is SPD.  The binding block is
%   a = argmin lc(a), and  lambda_min(C) D - omega L > 0  <=>  omega *
%   lambda_max(L, D) < lambda_min(C)  <=>  omega < omega_crit(Y).
%   For D = I this is simply lambda_min(C)/lambda_max(L).
%
%   Pinning omega to eta*omega_crit(Y) keeps the metric SPD at EVERY iterate
%   (eta < 1) while letting omega approach its admissible ceiling near Y*,
%   where the conditioning benefit is largest.  The map is scale-invariant
%   in (C, L), leaving only the benign safety factor eta to choose.
% ======================================================================
function w = omega_of_Y(Y, C, D, lcmin, eta)
    L = symm(Y' * C * Y * D);            % p-by-p multiplier (cheap)
    % Largest generalized eigenvalue of the pencil (L, D), D SPD.
    mu = eig(L, D);
    mu = real(mu);
    mumax = max(mu);
    if mumax <= 0
        % L is negative (semi)definite relative to D: H_Lag SPD for all
        % omega in [0,1); use the safety factor directly.
        w = min(max(eta, 0), 1 - 1e-12);
        return;
    end
    omega_crit = lcmin / mumax;
    w = eta * omega_crit;
    w = min(max(w, 0), 1 - 1e-12);       % clamp into [0,1)
end

% ======================================================================
% Riemannian gradient under the Lagrangian-preconditioned metric.
%
%   Solve  g_Y^omega(rgrad, Z) = <egrad, Z>  for all tangent Z, i.e.
%      rgrad = GY^{-1}( egrad - Y*S ),
%   with S = S' (p-by-p) enforcing tangency  sym(Y'*rgrad) = 0.
%
%   GY(Z) = 2 C Z D - 2 omega Z L,   L = sym(Y'CYD).
% ======================================================================
function rgrad = egrad2rgrad_Lag(Y, eg, C, D, omega, U, lc, p)

    L = symm(Y' * C * Y * D);

    % Apply GY^{-1} once to the Euclidean gradient.
    GinvEG = applyGinv_Lag(eg, U, lc, D, L, omega, p);

    % Build the small symmetric linear system for S:
    %   sym( Y' * GY^{-1}( Y*S ) ) = sym( Y' * GY^{-1}(egrad) ).
    % Vectorise over the p(p+1)/2 independent entries of symmetric S.
    [idx_i, idx_j] = symind(p);
    m = numel(idx_i);

    rhsMat = symm(Y' * GinvEG);
    b = zeros(m, 1);
    for c = 1:m
        b(c) = rhsMat(idx_i(c), idx_j(c));
    end

    A = zeros(m, m);
    for cc = 1:m
        S = zeros(p, p);
        S(idx_i(cc), idx_j(cc)) = 1;
        if idx_i(cc) ~= idx_j(cc)
            S(idx_j(cc), idx_i(cc)) = 1;
        end
        colMat = symm(Y' * applyGinv_Lag(Y * S, U, lc, D, L, omega, p));
        for rr = 1:m
            A(rr, cc) = colMat(idx_i(rr), idx_j(rr));
        end
    end

    svec = A \ b;

    S = zeros(p, p);
    for c = 1:m
        S(idx_i(c), idx_j(c)) = S(idx_i(c), idx_j(c)) + svec(c);
        if idx_i(c) ~= idx_j(c)
            S(idx_j(c), idx_i(c)) = S(idx_j(c), idx_i(c)) + svec(c);
        end
    end

    rgrad = applyGinv_Lag(eg - Y * S, U, lc, D, L, omega, p);

    % Numerical safeguard: project onto the tangent space of St(p,n).
    rgrad = rgrad - Y * symm(Y' * rgrad);
end

function rhess = ehess2rhess_Lag(Y, eg, eh, Z, C, D, omega, U, lc, p)
    lagHess = eh - Z * symm(Y' * eg);
    rhess = egrad2rgrad_Lag(Y, lagHess, C, D, omega, U, lc, p);
end

% ======================================================================
% WP10 fixed-base-point acceleration of the Lagrangian metric.
%
% egrad2rgrad_Lag spends essentially all of its time building the m-by-m
% tangency system A (m = p(p+1)/2), whose every column costs one
% applyGinv_Lag, i.e. n small p-by-p solves. A depends only on Y, omega and
% L -- never on the right-hand side. Inside hessianspectrum, Y is FIXED for
% all ~2001 applications, so A is rebuilt ~2001 times identically.
%
% prepare_lag_hessian builds A once and returns a solver whose Hessian
% closes over it. Nothing else changes: the same A\b solve, the same
% applyGinv_Lag calls for each new right-hand side, the same tangent
% safeguard. The returned operator is exact, not approximate.
%
% Measured at n=100, p=20 (rv_scratch/claude_wp10_validate.m):
%   per-application 1.714e-01 s -> 2.153e-03 s   (79.6x)
%   full spectrum   343.0 s (est.) -> 5.76 s     (59.5x)
%   cached vs uncached rgrad: max Frobenius difference exactly 0.
% ======================================================================
function solver = prepare_lag_hessian(solver, Y, C, D, lcmin, eta, U, lc, p)
    omega = omega_of_Y(Y, C, D, lcmin, eta);
    cache = make_lag_cache(Y, C, D, omega, U, lc, p);
    solver.M.egrad2rgrad = @(Yb, eg) egrad2rgrad_Lag_cached(Yb, eg, cache);
    solver.M.ehess2rhess = @(Yb, eg, eh, Z) ...
        egrad2rgrad_Lag_cached(Yb, eh - Z*symm(Yb'*eg), cache);
end

function cache = make_lag_cache(Y, C, D, omega, U, lc, p)
    cache.Y = Y;  cache.U = U;  cache.lc = lc;
    cache.D = D;  cache.omega = omega;
    cache.L = symm(Y' * C * Y * D);

    [idx_i, idx_j] = symind(p);
    cache.idx_i = idx_i;  cache.idx_j = idx_j;
    m = numel(idx_i);

    A = zeros(m, m);
    for cc = 1:m
        S = zeros(p, p);
        S(idx_i(cc), idx_j(cc)) = 1;
        if idx_i(cc) ~= idx_j(cc)
            S(idx_j(cc), idx_i(cc)) = 1;
        end
        colMat = symm(Y' * applyGinv_Lag(Y * S, U, lc, D, cache.L, omega, p));
        for rr = 1:m
            A(rr, cc) = colMat(idx_i(rr), idx_j(rr));
        end
    end
    % Deliberately NOT prefactorised with decomposition(): that object
    % auto-detects structure and may select a different factorisation than
    % mldivide does, which would perturb the last bits. The saving on a
    % 210-by-210 solve is ~3 ms against the ~170 ms already saved per
    % application, so bit-identity is worth far more than the factorisation.
    cache.A = A;
end

function rgrad = egrad2rgrad_Lag_cached(Y, eg, cache)
    % Guard: the cache is only valid at the base point it was built for.
    assert(isequal(Y, cache.Y), ...
        'prepare_lag_hessian cache used away from its base point.');

    U = cache.U;  lc = cache.lc;  D = cache.D;
    L = cache.L;  omega = cache.omega;
    p = size(Y, 2);

    GinvEG = applyGinv_Lag(eg, U, lc, D, L, omega, p);

    idx_i = cache.idx_i;  idx_j = cache.idx_j;
    m = numel(idx_i);
    rhsMat = symm(Y' * GinvEG);
    b = zeros(m, 1);
    for c = 1:m
        b(c) = rhsMat(idx_i(c), idx_j(c));
    end

    svec = cache.A \ b;

    S = zeros(p, p);
    for c = 1:m
        S(idx_i(c), idx_j(c)) = S(idx_i(c), idx_j(c)) + svec(c);
        if idx_i(c) ~= idx_j(c)
            S(idx_j(c), idx_i(c)) = S(idx_j(c), idx_i(c)) + svec(c);
        end
    end

    rgrad = applyGinv_Lag(eg - Y * S, U, lc, D, L, omega, p);
    rgrad = rgrad - Y * symm(Y' * rgrad);
end

% ----------------------------------------------------------------------
% Apply GY^{-1}:  solve  2 C Z D - 2 omega Z L = R  for Z.
%
%   In the C-eigenbasis (C = U diag(lc) U'),  with  Zt = U'*Z, Rt = U'*R,
%   each row a satisfies   Zt(a,:) * ( 2 lc(a) D - 2 omega L ) = Rt(a,:).
%   => n independent p-by-p right-hand-side solves.
% ----------------------------------------------------------------------
function Z = applyGinv_Lag(R, U, lc, D, L, omega, p)
    Rt = U' * R;                   % n-by-p
    n  = size(R, 1);
    Zt = zeros(n, p);
    twoOmegaL = 2 * omega * L;
    for a = 1:n
        Ma = 2 * lc(a) * D - twoOmegaL;   % p-by-p, SPD for omega in [0,1)
        % Solve  Zt(a,:) * Ma = Rt(a,:)   <=>   Ma' * x = Rt(a,:)'
        Zt(a, :) = (Ma' \ Rt(a, :)')';
    end
    Z = U * Zt;
end

% ----------------------------------------------------------------------
% Index list of the upper triangle (incl. diagonal) of a p-by-p matrix.
% ----------------------------------------------------------------------
function [ii, jj] = symind(p)
    ii = zeros(p*(p+1)/2, 1);
    jj = zeros(p*(p+1)/2, 1);
    c = 0;
    for i = 1:p
        for j = i:p
            c = c + 1;
            ii(c) = i;
            jj(c) = j;
        end
    end
end

% ======================================================================
% Symmetric square root of  (L^2 + delta*I)  for symmetric L (p-by-p).
% ======================================================================
function R = sqrtmL(L, delta, p)
    L = symm(L);
    [V, E] = eig(L);
    e = diag(E);
    R = V * diag(sqrt(e.^2 + delta)) * V';   % (L^2 + delta I)^{1/2}
    R = symm(R);
end

% ======================================================================
% Riemannian gradient under the left-and-right preconditioned metric.
%
%   Solve  g_Y^LR(rgrad, Z) = <egrad, Z>  for all tangent Z, i.e.
%      rgrad = GY^{-1}( egrad - Y*S ),
%   with S = S' (p-by-p) enforcing tangency  sym(Y'*rgrad) = 0.
%
%   GY(Z)       = C Z (L^2+delta I)^{1/2},  L = sym(Y'CYD).
%   GY^{-1}(R)  = C^{-1} R (L^2+delta I)^{-1/2}.
%
%   Tangency reduces to the generalized Sylvester equation
%      A S G + G S A = 2 sym( Y' GY^{-1}(egrad) ),
%   with  A = Y'C^{-1}Y  (spd) and  G = (L^2+delta I)^{-1/2}  (spd).
% ======================================================================
function rgrad = egrad2rgrad_LR(Y, eg, C, D, delta, U, lc, p)

    L = symm(Y' * C * Y * D);
    G = invsqrtmL(L, delta, p);            % (L^2 + delta I)^{-1/2}, spd

    % A = Y' C^{-1} Y, formed via the C-eigenbasis (no explicit inverse).
    Cinv_Y = U * ((U' * Y) ./ lc);         % C^{-1} Y
    A = symm(Y' * Cinv_Y);

    % Right-hand side  RHS = sym( Y' GY^{-1}(egrad) ).
    GinvEG = applyGinv_LR(eg, U, lc, G);
    RHS = symm(Y' * GinvEG);

    % Solve  A S G + G S A = 2 RHS  for symmetric S (Kronecker form).
    Ip2 = kron(G, A) + kron(A, G);         % since A,G symmetric
    svec = Ip2 \ (2 * RHS(:));
    S = reshape(svec, p, p);
    S = symm(S);

    % Riemannian gradient and a numerical tangency safeguard.
    rgrad = applyGinv_LR(eg - Y * S, U, lc, G);
    rgrad = rgrad - Y * symm(Y' * rgrad);
end

function rhess = ehess2rhess_LR(Y, eg, eh, Z, C, D, delta, U, lc, p)
    lagHess = eh - Z * symm(Y' * eg);
    rhess = egrad2rgrad_LR(Y, lagHess, C, D, delta, U, lc, p);
end

% ----------------------------------------------------------------------
% Apply GY^{-1}:  R |-> C^{-1} R G,  with  G = (L^2 + delta I)^{-1/2}.
%   C^{-1} is applied in its eigenbasis C = U diag(lc) U'.
% ----------------------------------------------------------------------
function Z = applyGinv_LR(R, U, lc, G)
    CinvR = U * ((U' * R) ./ lc);          % C^{-1} R
    Z = CinvR * G;                         % times (L^2+delta I)^{-1/2}
end

% ======================================================================
% Inverse symmetric square root of (L^2 + delta*I) for symmetric L.
% ======================================================================
function G = invsqrtmL(L, delta, p)
    L = symm(L);
    [V, E] = eig(L);
    e = diag(E);
    G = V * diag(1 ./ sqrt(e.^2 + delta)) * V';
    G = symm(G);
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
