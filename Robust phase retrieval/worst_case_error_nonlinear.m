% Define a function to compute E_{\lambda, \tilde C}(x) ---
function E = worst_case_error_nonlinear(x,  A, b, C, delta)
    F_x = F(x, A, b );
    num_vertices = 2^size(C, 2);
    Y = dec2bin(0:num_vertices-1) - '0';
    Y = fliplr(Y);
    all_z = 2 * delta * Y' - delta;
    objective_values = zeros(num_vertices, 1);
    for i = 1:num_vertices
        z_candidate = all_z(:, i);
        residual = F_x - C * z_candidate;
        objective_values(i) = 0.5 * sum(residual.^2);
    end
    E = max(objective_values);
end