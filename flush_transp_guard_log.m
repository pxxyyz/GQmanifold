function flush_transp_guard_log(log_path)
% FLUSH_TRANSP_GUARD_LOG  Write the deferred transport-guard counter CSV.
%
% function flush_transp_guard_log(log_path)
%
% generalfactory.m counts rejected transport inverses in the global
% GQ_TRANSP_GUARD_LOG but does not write them, so that file I/O stays out of
% the solver loop. Call this helper after the algorithm timers stop. A
% nonempty LOG_PATH is required; an empty registry writes nothing.
global GQ_TRANSP_GUARD_LOG
if strlength(string(log_path)) == 0 || isempty(GQ_TRANSP_GUARD_LOG),return;end
writetable(struct2table(GQ_TRANSP_GUARD_LOG),log_path);
end
