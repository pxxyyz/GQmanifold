% example_trace.m
%
% Trace minimization  min { tr(X'*A*X) : X'*Q*X = P }  on Gq(P,Q), run in the
% three configurations that one factory covers: symmetric positive definite
% (P,Q), skew-symmetric (P,Q), and symmetric indefinite Q with P = diag(+/-1).
% Each configuration is solved with Riemannian conjugate gradient and with
% Riemannian trust regions, over every retraction admissible in that
% configuration, and the Riemannian gradient norm and the feasibility error
% ||X'*Q*X - P||_F are recorded along the iterate sequence.
%
% Reproduces the trace-minimization figure and tables of Section 5.2 of the
% paper "Generalized Quadratic Matrix Manifolds: Geometry and Computations".
%
% Seed rng(1), set once at the top. Dimensions n = 100, p = 10.
%
% Run from this directory, with Manopt on the path:
%   >> example_trace
%
% Outputs, all written to results/ :
%   trace_RCG_n_100_p_10.pdf, trace_RTR_n_100_p_10.pdf  convergence figures
%   trace_symm_n_100_p_10.csv, trace_skew_n_100_p_10.csv,
%   trace_indef_n_100_p_10.csv                          per-configuration tables
%   trace_fixture_<case>_n_100_p_10.mat                 the realized instance

clear; close all; clc; rng(1);
disp("##############################");
disp("# trace minimization problem #");
disp("##############################");

%% Problem dimensions and (fixed, undisplayed) conditioning of the cost matrix
n = 100; p = 10;
kappa = 100;   % condition number of the SPD cost matrix A; not shown in figures

%% Cases. Gq(P,Q) covers, in one factory:
%   symm  : symmetric SPD P,Q              (generalized Stiefel-type)
%   skew  : skew-symmetric P,Q             (symplectic Stiefel-type)
%   indef : symmetric INDEFINITE Q, P = J=diag(+/-1) with J^2 = I
%           -> this is the indefinite Stiefel manifold iSt_{A,J} = Gq(J,A).
case_list  = {'symm', 'skew', 'indef'};
case_label = {'Symmetric case', 'Skew-symmetric case', 'Indefinite case'};

%% Solvers (column blocks). Retractions are chosen per case below.
method_list = {'R-CG', 'R-TR'};
method_name = {'RCG', 'RTR'};
lenc = numel(case_list);
lenm = numel(method_list);
maxr = 3;                 % at most three retractions in any case

%% Shared SPD cost matrix A (common spectrum across cases)
U_seed = sign(rand(n, n) - 0.5);
[U_star, ~, ~] = svds(U_seed, n);
sigma = linspace(1, 1/kappa, n);
A = U_star*diag(sigma)*U_star';

prob.cost  = @(X) trace(X'*A*X);
prob.egrad = @(X) 2*A*X;
prob.ehess = @(X, Xdot) 2*A*Xdot;

%% Solver options (record per-iterate feasibility via statsfun)
opt.verbosity   = 0;
opt.maxiter     = 5e3;
opt.tolgradnorm = 1e-8;

%% Output directory (hoisted: the per-case fixture archive writes here)
output_dir = 'results/';
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

%% Run all configurations: case x solver x retraction
out_infos  = cell(lenc, lenm, maxr);
caseretr_n = cell(lenc, 1);          % retraction names used in each case
T_rows     = {};
for ic = 1:lenc
    switch case_list{ic}
        case 'symm'
            symm = @(D) sqrtm(D*D') + 1e-6*eye(size(D));
            Q = symm(randn(n)); P = symm(randn(p));
        case 'skew'
            skew = @(D) (D - D')/2;
            Q = skew(randn(n)); P = skew(randn(p));
        case 'indef'
            % symmetric indefinite Q with inertia (npos,nneg), npos,nneg >= p/2
            Uq = orth(randn(n));
            npos = round(0.6*n); nneg = n - npos;
            dQ = [linspace(1, 10, npos), -linspace(1, 10, nneg)];
            Q = Uq*diag(dQ)*Uq'; Q = (Q + Q')/2;
            % P = J = diag(I_{p/2}, -I_{p/2}) satisfies J^2 = I
            P = diag([ones(1, p/2), -ones(1, p/2)]);
    end

    M = generalfactory(Q, P);
    M.feasi = @(X) norm(X'*Q*X - P, 'fro');

    % All three retractions apply to every case (the polar retraction is now
    % well defined for the indefinite case via the real principal square root).
    rlist = {M.retr_cayley, M.retr_pol, M.retr_qgeo};
    rname = {'cay', 'pol', 'qgeo'};
    caseretr_n{ic} = rname;

    X0 = M.rand();
    disp([case_label{ic}, ': feasi(X0) = ', num2str(M.feasi(X0))]);
    % Archive the realized instance so the table can be recomputed
    % without rerunning the generator. Save only; nothing is read back, so
    % the trajectory is unchanged.
    fixture = struct('case', case_list{ic}, 'n', n, 'p', p, 'kappa', kappa, ...
                     'seed', 1, 'A', A, 'P', P, 'Q', Q, 'X0', X0, ...
                     'feasi_X0', M.feasi(X0));
    save(fullfile(output_dir, sprintf('trace_fixture_%s_n_%d_p_%d.mat', ...
         case_list{ic}, n, p)), '-struct', 'fixture');

    opt.statsfun = statsfunhelper('feasi', @(X) M.feasi(X));
    case_opt = opt;
    if strcmp(case_list{ic}, 'indef')
        % Near the floating-point cost floor on the non-compact indefinite
        % case, rho regularization can accept genuine cost increases and
        % repeatedly kick RTR out of its fast local convergence regime.
        case_opt.rho_regularization = 0;
    end

    prob.M = M;
    for im = 1:lenm
        for ir = 1:numel(rlist)
            prob.M.retr = rlist{ir};
            % Pick the matching vector transport explicitly. (Do not use
            % M.selecttransp: it is a nested-function closure that reads the
            % factory's default retraction, not prob.M.retr, so it always
            % returns the Cayley transport.)
            switch rname{ir}
                case 'cay';  prob.M.transp = M.transp_cayley;
                case 'pol';  prob.M.transp = M.transp_pol;
                case 'qgeo'; prob.M.transp = @(x1, x2, d) M.proj(x2, d);
            end
            disp(['  ', case_label{ic}, ' | ', method_name{im}, ' | ', rname{ir}]);
            [out, ~, info]    = callmethod(method_list{im}, prob, X0, case_opt);
            out_infos{ic, im, ir} = info;
            % Classify the actual Manopt exit. Manopt does not export a
            % reason code, so it is reconstructed from the three active criteria.
            stop_reason = 'gradtol';
            if info(end).gradnorm > case_opt.tolgradnorm
                if info(end).iter >= case_opt.maxiter
                    stop_reason = 'maxiter';
                else
                    stop_reason = 'minstepsize';
                end
            end
            if isfield(info(end), 'stepsize'), last_step = info(end).stepsize;
            else, last_step = NaN; end
            T_rows{end+1} = {case_list{ic}, method_name{im}, rname{ir}, ...
                prob.cost(out), info(end).gradnorm, M.feasi(out), ...
                info(end).iter, info(end).time, stop_reason, last_step}; %#ok<SAGROW>
        end
    end
end

%% Summary table
varNames = ["case", "method", "retr", "fval", "gradf", "feasi", "iter", "time", "stop_reason", "stepsize"];
T = cell2table(vertcat(T_rows{:}), 'VariableNames', varNames);
disp(T);

%% Plot: one figure per solver. In each figure the columns are the three cases
%  (symmetric, skew-symmetric, indefinite) and the rows are the Riemannian
%  gradient norm and the feasibility, with one colored curve per retraction.
%  Feasibility shares one axis across cases so the constraint residual is seen
%  to stay at the roundoff floor. The two solvers (RCG, RTR) are saved as
%  separate files.
close all;

% Times New Roman if available, otherwise a safe fallback.
FONT_NAME = 'Times New Roman';
try
    if ~any(strcmpi(listfonts, FONT_NAME)); FONT_NAME = 'Helvetica'; end
catch
    FONT_NAME = 'Helvetica';
end
FS_TICK = 10; FS_LABEL = 12; FS_TITLE = 13; FS_LEGEND = 12;
LINE_WIDTH = 1.4; MARKER_SIZE = 5; MARKER_NUM = 6;
TOL = opt.tolgradnorm;

GRAD_LIM  = [1e-12, 1e1];  GRAD_TICKS  = 10.^(-12:3:0);
FEASI_LIM = [1e-16, 1e-8]; FEASI_TICKS = 10.^(-16:2:-8);

qty   = {'gradnorm', 'feasi'};
qlab  = {'$\|\mathrm{grad}\,f(X_k)\|$', '$\|X_k^\top Q X_k - P\|_F$'};
qlim  = {GRAD_LIM, FEASI_LIM};
qtick = {GRAD_TICKS, FEASI_TICKS};
case_title = {'Symmetric positive-definite', 'Skew-symmetric', 'Symmetric indefinite'};

for im = 1:lenm
    % iteration range per case for this solver (both rows of a column share it)
    xmax_case = zeros(lenc, 1);
    for ic = 1:lenc
        for jr = 1:numel(caseretr_n{ic})
            xmax_case(ic) = max(xmax_case(ic), ...
                max([out_infos{ic, im, jr}.iter]));
        end
    end

    fig = figure('Color', 'white', 'Units', 'inches', 'Position', [0 0 9.2 5.0]);
    tl  = tiledlayout(fig, 2, lenc, 'TileSpacing', 'compact', 'Padding', 'compact'); %#ok<NASGU>

    hleg = gobjects(maxr, 1); leg_names = {};
    for iq = 1:2
        for ic = 1:lenc
            ax = nexttile; hold(ax, 'on'); box(ax, 'on');
            rname = caseretr_n{ic}; nret = numel(rname);
            for jr = 1:nret
                [col, mk] = getstyle(rname{jr});
                info = out_infos{ic, im, jr};
                x = [info.iter];
                y = max([info.(qty{iq})], realmin);
                midx = unique(round(linspace(1, numel(x), min(numel(x), MARKER_NUM))));
                h = semilogy(ax, x, y, '-', 'Color', col, ...
                    'LineWidth', LINE_WIDTH, 'Marker', mk, ...
                    'MarkerSize', MARKER_SIZE, 'MarkerIndices', midx);
                if iq == 1 && ic == 1
                    hleg(jr) = h; leg_names{jr} = retrname_full(rname{jr}); %#ok<SAGROW>
                end
            end
            set(ax, 'YScale', 'log', 'FontName', FONT_NAME, 'FontSize', FS_TICK, ...
                'YGrid', 'on', 'GridAlpha', 0.12, 'YMinorTick', 'off');
            set(ax, 'YLim', qlim{iq}, 'YTick', qtick{iq});
            if xmax_case(ic) > 0, set(ax, 'XLim', [0, xmax_case(ic)]); end
            if iq == 1
                % stopping-tolerance reference line (kept out of the legend)
                xl = get(ax, 'XLim');
                plot(ax, xl, [TOL, TOL], ':', 'Color', [0.45 0.45 0.45], ...
                    'LineWidth', 1.0, 'HandleVisibility', 'off');
                title(ax, case_title{ic}, 'FontName', FONT_NAME, 'FontSize', FS_TITLE);
            end
            if ic == 1
                ylabel(ax, qlab{iq}, 'Interpreter', 'latex', 'FontSize', FS_LABEL);
            end
            if iq == 2
                xlabel(ax, 'Iteration', 'FontName', FONT_NAME, 'FontSize', FS_LABEL);
            end
        end
    end

    lgd = legend(hleg(1:numel(leg_names)), leg_names, 'Orientation', 'horizontal', ...
        'Interpreter', 'latex', 'FontName', FONT_NAME, 'FontSize', FS_LEGEND);
    try
        lgd.Layout.Tile = 'south';
    catch
        set(lgd, 'Location', 'southoutside');
    end

    fpdf = fullfile(output_dir, sprintf('trace_%s_n_%d_p_%d.pdf', ...
        method_name{im}, n, p));
    exportgraphics(fig, fpdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
end

% Per-case data tables (both solvers, all retractions).
for ic = 1:lenc
    sub = T(strcmp(T.case, case_list{ic}), :);
    writetable(sub, fullfile(output_dir, ...
        sprintf('trace_%s_n_%d_p_%d.csv', case_list{ic}, n, p)));
end

%% Fixed color/marker per retraction (consistent across all panels)
function [col, mk] = getstyle(name)
switch name
    case 'cay';  col = [0.00 0.45 0.74]; mk = 'o';    % blue
    case 'pol';  col = [0.85 0.33 0.10]; mk = 's';    % vermilion
    case 'qgeo'; col = [0.47 0.67 0.19]; mk = '^';    % green
    otherwise;   col = [0.00 0.00 0.00]; mk = '.';
end
end

%% Full retraction names for the legend
function s = retrname_full(name)
switch name
    case 'cay';  s = 'Cayley';
    case 'pol';  s = 'polar';
    case 'qgeo'; s = 'quasi-geodesic';
    otherwise;   s = name;
end
end

%%
function [x, cost, info] = callmethod(method, prob, X0, opt)
switch upper(method)
    case 'R-TR'
        [x, cost, info] = trustregions(prob, X0, opt);
    case 'R-BB'
        [x, cost, info] = barzilaiborwein(prob, X0, opt);
    case 'R-CG'
        [x, cost, info] = conjugategradient(prob, X0, opt);
    case 'R-SD'
        [x, cost, info] = steepestdescent(prob, X0, opt);
    case 'R-BFGS'
        [x, cost, info] = rlbfgs(prob, X0, opt);
    otherwise
        error(['Unknown method type. ' ...
            'Should be R-TR, R-BB, R-CG, R-SD or R-BFGS.']);
end
end
