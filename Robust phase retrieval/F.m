% F(x) = max(Ax - b, d)
function F_x = F(x, A, b)
   F_x = (A * x).^2 - b;
% F_x = sin(A * x) - b;
end