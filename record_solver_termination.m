function result = record_solver_termination(problem,x,options,info)
% RECORD_SOLVER_TERMINATION  Apply Manopt's stopping rules to an endpoint.
%
% function result = record_solver_termination(problem, x, options, info)
%
% Manopt returns the iterate history but not a reason code. This helper calls
% Manopt's own stoppingcriterion on the returned endpoint, labels the result,
% and keeps the options that determine the label. Called outside the algorithm
% timers by example_symm.m and example_skew.m.
[stop,reason]=stoppingcriterion(problem,x,options,info,numel(info));
if stop
 switch stop
  case 1,label='cost_tolerance';
  case 2,label='gradient_tolerance';
  case 3,label='time_limit';
  case 4,label='iteration_cap';
  case 6,label='user_stopping_rule';
  otherwise,label='standard_stopping_rule';
 end
elseif abs(info(end).stepsize)<options.minstepsize
 stop=true;label='minimum_stepsize';
 reason=sprintf('Last stepsize smaller than minimum allowed; options.minstepsize = %g.',options.minstepsize);
else
 label='unclassified';reason='No recorded stopping rule matches endpoint';
end
result=struct('stop',logical(stop),'label',label,'reason',reason,...
 'iter',info(end).iter,'gradnorm',info(end).gradnorm,'stepsize',info(end).stepsize,...
 'maxiter',options.maxiter,'tolgradnorm',options.tolgradnorm,'minstepsize',options.minstepsize,...
 'beta_type',options.beta_type,'orth_value',options.orth_value,'linesearch',func2str(options.linesearch));
end
