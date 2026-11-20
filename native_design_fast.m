function [result, state] = native_design_fast(family,EC,ED)
% NATIVE_DESIGN_FAST  Exact affine design model with native sparse solvers.
%
% function [result, state] = native_design_fast(family, EC, ED)
%
% FAMILY is 'symm' or 'skew'. EC and ED are the canonical spectra used by
% example_symm.m and example_skew.m. The first stage is an affine linear
% program solved by linprog with the dual-simplex-highs algorithm; the second
% stage minimizes ||x||^2 + ||y||^2 subject to the first-stage spread plus a
% relative slack, and is solved by quadprog. The relative slack ladder is
% 1e-9, 1e-8, ..., 1e-4 times max(1, |LP optimum|); the first successful
% stage is accepted.
%
% RESULT carries the design weights and solver records; STATE carries the
% sparse matrices, multipliers, and solutions needed by
% native_design_diagnostics.m. No random numbers are drawn, so the caller's
% random stream is unchanged.
t0 = tic;
EC=EC(:);ED=ED(:);
if strcmp(family,'symm'),nu=1;elseif strcmp(family,'skew'),nu=2;else,error('native_design_fast:family','Unknown family ''%s''; use ''symm'' or ''skew''.',family);end
n=nu*numel(EC);p=nu*numel(ED);
q=p/nu; m=n/nu;
[I,J]=meshgrid(1:q,1:q);
Mminus=((1./EC(I)-1./EC(J)).*(ED(J)-ED(I)))/2;
logMminus=log(Mminus);
if nu==1, mask=logMminus>-inf; else, mask=triu(logMminus>-inf,1); end
[i,j]=find(mask); a=logMminus(mask);
F=sparse((1:numel(i))',i,-1,numel(i),m)+sparse((1:numel(j))',j,-1,numel(j),m);
if nu==2
 Mplus=((1./EC(I)+1./EC(J)).*(ED(J)+ED(I)))/2;
 lm=log(Mplus); mask=triu(lm>-inf,1); [i,j]=find(mask);
 F=[F;sparse((1:numel(i))',i,-1,numel(i),m)+sparse((1:numel(j))',j,-1,numel(j),m)]; a=[a;lm(mask)];
 F=[F;sparse((1:q)',(1:q)',-2,q,m)];a=[a;log(2.*ED./EC(1:q))];
end
[I,J]=meshgrid(1:q,q+1:m);
Nminus=(1./EC(J)-1./EC(I)).*ED(I); ln=log(Nminus);mask=ln>-inf;[i,j]=find(mask);
F=[F;sparse((1:numel(i))',q+i,1,numel(i),m)+sparse((1:numel(j))',j,-1,numel(j),m)];a=[a;ln(mask)];
if nu==2
 Nplus=(1./EC(J)+1./EC(I)).*ED(I);ln=log(Nplus);mask=ln>-inf;[i,j]=find(mask);
 F=[F;sparse((1:numel(i))',q+i,1,numel(i),m)+sparse((1:numel(j))',j,-1,numel(j),m)];a=[a;ln(mask)];
end
[~,~,idxED]=unique(ED,'stable');[i,j]=find(triu(idxED==idxED',1));
E=sparse((1:numel(i))',i,1,numel(i),m+2)+sparse((1:numel(j))',j,-1,numel(j),m+2);
be=zeros(numel(i),1);nr=numel(a);
A=[F,-ones(nr,1),zeros(nr,1);-F,zeros(nr,1),ones(nr,1)];b=[-a;a];
f=[zeros(m,1);1;-1];H=2*spdiags([ones(m,1);0;0],0,m+2,m+2);
assembly=toc(t0);
option_clock=tic;
lpopt=optimoptions('linprog','Display','off','Algorithm','dual-simplex-highs','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-9);
qpopt=optimoptions('quadprog','Display','off','Algorithm','interior-point-convex','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-9,'MaxIterations',200);
option_time=toc(option_clock);
t=tic;[z1,v1,ef1,out1,lam1]=linprog(f,A,b,E,be,[],[],lpopt);lpwall=toc(t);

assert(ef1==1,'Native LP did not return an optimal result');
A2=[A;f'];
qpwall=0;attempts=0;attempt_records=repmat(struct('slack_scale',NaN,'slack',NaN,'exitflag',NaN,'objective',NaN,'output',struct()),0,1);
for slack_scale=[1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
 attempts=attempts+1;slack=slack_scale*max(1,abs(v1));b2=[b;v1+slack];
 t=tic;[z2,v2,ef2,out2,lam2]=quadprog(H,zeros(m+2,1),A2,b2,E,be,[],[],[],qpopt);qpwall=qpwall+toc(t);
 attempt_records(attempts)=struct('slack_scale',slack_scale,'slack',slack,'exitflag',ef2,'objective',v2,'output',out2);
 if ef2==1,break;end
end
assert(ef2==1,'Native secondary QP did not return an optimal result');
program_wall=toc(t0);
result=struct('family',family,'n',n,'p',p,'modes',nr,'equalities',size(E,1),...
 'assembly',assembly,'options',option_time,'lp_time',lpwall,'qp_time',qpwall,...
 'program_wall',program_wall,'timed_block_sum',assembly+option_time+lpwall+qpwall,...
 'lp_value',v1,'qp_value',v2,'slack',slack,'slack_scale',slack_scale,'stage2_attempts',attempts,...
 'lp_exitflag',ef1,'qp_exitflag',ef2,'lp_output',out1,'qp_output',out2,...
 'attempt_records',attempt_records,...
 'x',z2(1:q),'y',z2(q+1:m),'EP',exp(z2(1:q)),'EQ',[ones(q,1);exp(z2(q+1:m))],...
 'lp_settings',struct('Algorithm','dual-simplex-highs','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-9),...
 'qp_settings',struct('Algorithm','interior-point-convex','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-9,'MaxIterations',200));
state=struct('F',F,'a',a,'A',A,'b',b,'E',E,'be',be,'f',f,'H',H,...
 'A2',A2,'b2',b2,'z1',z1,'z2',z2,'lambda1',lam1,'lambda2',lam2);
end
