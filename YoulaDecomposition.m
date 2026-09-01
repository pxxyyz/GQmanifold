function [U, S] = YoulaDecomposition(Q)
% YOULADECOMPOSITION  Real Youla (skew-symmetric Schur) decomposition.
%
% function [U, S] = YoulaDecomposition(Q)
%
% For a real skew-symmetric matrix Q of even order, returns the real canonical
% form S, block diagonal with 2-by-2 skew blocks carrying the moduli of the
% purely imaginary eigenvalues of Q in descending order, together with the
% transformation U that realizes it. Used by the skew-symmetric branch of
% generalfactory.m and by example_skew.m.
%
[n1, n2] = size(Q);
if n1 ~= n2
    error('The input is not a square matrix.')
elseif mod(n1, 2)
    error('The input is not a even-dimensional matrix.')
end
n = n1/2;
J2 = [0, 1; -1, 0];
Jhat = kron(eye(n), J2);
[UQ, EQ] = eig(Q);
eigQ = diag(imag(EQ));
[~, indaa] = sort(eigQ(1:2:end), 'descend');
ind = 1:2:length(eigQ); 
ind = ind(indaa);
ind = vec([ind; ind+1]);
eigQ = eigQ(ind);
UQ = UQ(:,ind);
% S = diag(abs(eigQ).^0.5)*Jhat*diag(abs(eigQ).^0.5);
S = diag(abs(eigQ))*Jhat;
[US, ~] = eig(S);
U = UQ*US';
if norm(imag(U)) < 1e-5
    U = real(U);
end
for k = 1:n
    c1 = 2*k - 1;
    c2 = 2*k;
    a = U(:, c1);
    b = U(:, c2);
    w = a.^2 + b.^2;
    [~, i] = max(w);
    theta = atan2(b(i), a(i));
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    U(:, [c1 c2]) = U(:, [c1 c2])*R;
end
end
