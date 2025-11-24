function phi_x = phi(A,b,C,x_rls,num_vertices,y,all_z)     
% --- Step 1: Enumerate to solve inner problem max_z 1/2 ||F(x_rls) - (y - C*z)||^2 ---
    F_x = F(x_rls, A, b);
    residual_base = F_x - y;
    phi_x  = 0;
    for i = 1:num_vertices
        z_candidate = all_z(:, i);
        residual = residual_base + C * z_candidate;
        obj = 0.5 * sum(residual.^2);
        if obj >= phi_x
            phi_x = obj;
        end
    end
