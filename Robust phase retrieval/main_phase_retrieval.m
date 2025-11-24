%% Nonlinear Least Squares: LS, LASSO, RLS Robustness Comparison (Multiple delta_base)
%  Goal: Compare the robustness of LS, LASSO, and RLS on a phase retrieval problem.
%        Show median results over multiple trials for multiple delta_base values.

clear; close all; clc;

% --- 1.1 Define Problem Dimensions ---
m = 300; % Number of measurements (dimension of F(x))
n = 100; % Signal dimension (dimension of x)
k = 30; % Sparsity level (approximately 1/3)
num_trials = 10;

% --- 1.5 Generate Noise Matrix C ---
n_noise = 6; % Number of noise patterns

% --- 1.6 Define delta evaluation range ---
delta_eval_min = 0;
delta_eval_max = 0.2;
num_delta_points = 20;
delta_eval = linspace(delta_eval_min, delta_eval_max, num_delta_points + 1);

% --- Define multiple delta_base values ---
delta_base_values = [1e-1, 1e-2, 1e-3]; % [10^{-1}, 10^{-2}, 10^{-3}]
num_delta_bases = length(delta_base_values);

% --- Initialize storage arrays ---
% Store Delta curves for all trials for each delta_base
Delta_ls_all = zeros(num_delta_points + 1, num_trials, num_delta_bases);
Delta_lasso_all = zeros(num_delta_points + 1, num_trials, num_delta_bases);


%% 4. Run Experiments ---

for idx = 1:num_delta_bases
    delta_base = delta_base_values(idx);
    fprintf('\n--- Running experiments for delta_base = %.2e ---\n', delta_base);
%     C = generate_acute_matrix(m,n_noise);
    C = generateObtuseMatrix(m,n_noise);

    for trial = 1:num_trials
        fprintf('    Trial %d out of %d\n', trial, num_trials);
        
        %% 1.2 Generate true sparse signal x_true ---
        x_true = zeros(n, 1);
        nonzero_indices = randperm(n, k);
        x_true(nonzero_indices) = 2 * randn(k, 1);                     

        %% 1.3 Generate measurement matrix A and offset vector b ---
        A = randn(m, n); 
        b_true = A * x_true; 

        %% 1.6 Generate observations y ---
        % Generate true noise vector z_true
        z_true = delta_base * (2 * rand(n_noise, 1) - 1);
        % Measured values (with structured noise)
        b = b_true + C * z_true;
        % Add small Gaussian noise
        b = b + 0.05 * randn(size(b));
        b = b.^2; % Apply nonlinearity
        
        %% 3. Apply Standard Least Squares (LS) model
        fprintf('    Solving Standard Least Squares (LS)...\n');
        objective_ls = @(x) 0.5 * sum((F(x, A, b)).^2);
        x0 = randn(n, 1);
        [x_ls, ~] = fminunc(objective_ls, x0);

        %% 4. Apply LASSO model
        fprintf('    Solving LASSO model...\n');
        objective_lasso = @(x) 0.5 * sum((F(x, A, b)).^2) + 10 * delta_base * norm(x, 1);
        [x_lasso, ~] = fminunc(objective_lasso, x0);

        %% 5. Apply Robust Least Squares (RLS) model
        fprintf('    Starting Robust Least Squares (RLS)...\n');
        objective_rls = @(x) worst_case_error_nonlinear(x, A, b, C, delta_base);
        [x_rls, ~] = fminunc(objective_rls, x0);

        %% 6. Compute robustness metric Delta vs. delta ---
        for i = 1:length(delta_eval)
            current_delta = delta_eval(i);
            E_ls = worst_case_error_nonlinear(x_ls, A, b, C, current_delta);
            E_lasso = worst_case_error_nonlinear(x_lasso, A, b, C, current_delta);
            E_rls = worst_case_error_nonlinear(x_rls, A, b, C, current_delta);
            Delta_ls_all(i, trial, idx) = E_ls - E_rls;
            Delta_lasso_all(i, trial, idx) = E_lasso - E_rls;
        end
    end
end

%% 7. Compute Median and Plot ---
% Calculate median
Delta_ls_median = median(Delta_ls_all, 2, 'omitnan'); % Median over 2nd dimension (trials)
Delta_lasso_median = median(Delta_lasso_all, 2, 'omitnan'); % Median over 2nd dimension (trials)

% Plot median curves
figure('Position', [100, 100, 1000, 600]);

% Define uniform line styles
colors = ['b', 'g', 'r']; % Blue, Green, Red
markers = {'o', 'x', '*'}; % Circle, X, Star

% Subplot 1: Δ_ls
subplot(1, 2, 1);
hold on;
% Store all plot handles
plot_handles_ls = [];
for idx = 1:num_delta_bases
    exponent = floor(log10(delta_base_values(idx)));
    h = plot(delta_eval, squeeze(Delta_ls_median(:, idx)), '-', 'Color', colors(idx), 'Marker', markers{idx}, ...
         'LineWidth', 1.8, 'DisplayName', sprintf('\\delta=10^{%d}', exponent));
    plot_handles_ls(end+1) = h; % Add handle to array
end
h_yline1 = yline(0, 'k--', 'LineWidth', 1);
xlabel('\lambda', 'FontSize', 16, 'FontName', 'Times New Roman');
ylabel('\Delta_{ls}', 'FontSize', 16, 'FontName', 'Times New Roman');
% Create legend with only plot handles, order maintained by loop
legend(plot_handles_ls, 'Location', 'best', 'FontSize', 14);
grid on;

% Subplot 2: Δ_lasso
subplot(1, 2, 2);
hold on;
% Store all plot handles
plot_handles_lasso = [];
for idx = 1:num_delta_bases
    exponent = floor(log10(delta_base_values(idx)));
    h = plot(delta_eval, squeeze(Delta_lasso_median(:, idx)), '-', 'Color', colors(idx), 'Marker', markers{idx}, ...
         'LineWidth', 1.8, 'DisplayName', sprintf('\\delta=10^{%d}', exponent));
    plot_handles_lasso(end+1) = h; % Add handle to array
end
h_yline2 = yline(0, 'k--', 'LineWidth', 1);
xlabel('\lambda', 'FontSize', 16, 'FontName', 'Times New Roman');
ylabel('\Delta_{lasso}', 'FontSize', 16, 'FontName', 'Times New Roman');
% Create legend with only plot handles, order maintained by loop
legend(plot_handles_lasso, 'Location', 'best', 'FontSize', 14);
grid on;

% Adjust layout
% sgtitle('Median Robustness Gain of RLS vs. Noise Level \lambda (Nonlinear Model)');

%% 8. Print Key Results ---
fprintf('\n--- Final Results (Median over %d trials) ---\n', num_trials);
fprintf('Evaluation delta range: [%.2f, %.2f]\n', delta_eval_min, delta_eval_max);
for idx = 1:num_delta_bases
    fprintf('For delta_base = %.2e:\n', delta_base_values(idx));
    fprintf('  Final Median Delta_ls at lambda=%.2f: %.4f\n', delta_eval(end), Delta_ls_median(end, idx));
    fprintf('  Final Median Delta_lasso at lambda=%.2f: %.4f\n', delta_eval(end), Delta_lasso_median(end, idx));
end