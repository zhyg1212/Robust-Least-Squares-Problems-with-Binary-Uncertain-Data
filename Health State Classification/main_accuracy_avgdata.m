%% Accuracy Comparison Plot: LS, LASSO, RLS (Full Coverage) and RLS (Partial Coverage)
%  Author: [Your Name]
%  Goal: Generate accuracy comparison plots for four models under different noise levels.
%  Uses CHEN and KELLY's explicit inner solution method
%  Randomly group data with the same labels, with group sizes of 100 and 1000 dimensions
%  Same number of samples in each group, data is averaged.
%  Data processing steps:
%  1. Read individual user data and extract matrix of total_samples*28;
%  2. Remove rows where labels are not 0 or 1, this matrix is used as test matrix A, 
%     and its corresponding labels are used as test b;
%  3. Count how many rows remain in current sample, divide by 100 or 1000 and round down, 
%     to get samples per group
%  4. According to calculated samples per group, perform grouping without replacement sampling, 
%     and take average, resulting 100 * 28 and 1000 * 28 matrices are our training matrix A. 
%     Corresponding labels are denoted as b.

clear; close all; clc;

r = 10000; % or 3000, number of rows in matrix A

for idx = 1:10
[merged_data, success] = load_and_merge_user_data(idx);
noise_ratios = 0:0.05:0.5;
num_ratios = length(noise_ratios);

if success
    % 1. Add user number to data name
    data_name = sprintf('merged_data_user_%d', idx);
    assignin('base', data_name, merged_data); % Store data in workspace with user number
    
    % Filter samples with status 0 and 1
    status_vector = merged_data.status; % Assume column name is 'status'
    idx_s = find(status_vector == 1 | status_vector == 0);
    merged_data_filtered = merged_data(idx_s,:); % Filtered data for testing
    
    columns_to_remove = {'time', 'activity', 'status', 'gait_type', 'motion_type'};
    existing_columns_to_remove = intersect(columns_to_remove, merged_data_filtered.Properties.VariableNames, 'stable');
    
    % --- Step 2: Generate test matrix A_test and b_test ---
    A_test = merged_data_filtered(:, ~ismember(merged_data.Properties.VariableNames, existing_columns_to_remove));
    A_test = table2array(A_test); % Convert to numerical matrix
    b_test = merged_data_filtered.status;
    
    % --- Step 1 & 3 & 4: Generate training matrix A and b_clean ---
    % Extract feature matrix and labels for training
    train_features = merged_data_filtered(:, ~ismember(merged_data.Properties.VariableNames, existing_columns_to_remove));
    train_features = table2array(train_features);
    train_labels = merged_data_filtered.status;
    
    % Define two target dimensions
    
    % Initialize variables for storing final training data
    A = []; 
    b_clean = [];
    
        num_groups = r;
        total_samples = size(train_features, 1);
        
        % --- Step 3: Calculate samples per group ---
        samples_per_group = floor(total_samples / num_groups);
        if samples_per_group == 0
            error('Insufficient samples to divide into %d groups.', num_groups);
        end
        
        % Actual total samples used (divisible by number of groups)
        total_used = samples_per_group * num_groups;
        
        % --- Step 4: Sample without replacement and take average ---
        % Randomly shuffle indices
        shuffled_idx = randperm(total_samples);
        % Take first total_used indices
        used_idx = shuffled_idx(1:total_used);
        
        % Initialize training data for this dimension
        A_dim = zeros(num_groups, size(train_features, 2));
        b_dim = zeros(num_groups, 1);
        
        for g = 1:num_groups
            % Calculate current group index range
            start_idx = (g-1) * samples_per_group + 1;
            end_idx = g * samples_per_group;
            group_idx = used_idx(start_idx:end_idx);
            
            % Average samples within group
            A_dim(g, :) = mean(train_features(group_idx, :), 1);
            % Average labels within group (or take median/mode, here use mean then round)
            b_dim(g) = round(mean(train_labels(group_idx)));
        end
        A = A_dim;
        b_clean = b_dim;
    
else
    fprintf('Unable to load data for user %d.\n', idx);
end

[num_samples, m] = size(A); % m = 784


%% 2. Define range of error label ratios
% X-axis: ratio of error labels


% Initialize accuracy storage vectors
accuracy_ls = zeros(1, num_ratios);
accuracy_lasso = zeros(1, num_ratios);
accuracy_rls_full = zeros(1, num_ratios);
accuracy_rls_partial = zeros(1, num_ratios); % For RLS with partial coverage

test_accuracy_ls = zeros(1, num_ratios);
test_accuracy_lasso = zeros(1, num_ratios);
test_accuracy_rls_full = zeros(1, num_ratios);
test_accuracy_rls_partial = zeros(1, num_ratios); % For RLS with partial coverage

%% 3. Iterate through different noise ratios, calculate accuracy for all models
lambda_lasso = 1; % LASSO regularization parameter

for i = 1:num_ratios
    noise_ratio = noise_ratios(i);
    fprintf('Processing noise ratio = %.2f...\n', noise_ratio);
    
    %% 3.1 Apply structured noise to labels (Adversarial Label Noise)
    num_noise = round(noise_ratio * num_samples);
    % Randomly select samples to be actually flipped
    true_flip_indices = randperm(num_samples, num_noise);
    
    % --- Generate observed labels with noise b_obs (for LS and LASSO) ---
    b_obs = b_clean;
    b_obs(true_flip_indices) = 1 - b_obs(true_flip_indices); % Flip labels
    
    %% 3.2 Train and evaluate LS model
    x_ls = A \ b_obs;
    pred_ls = A * x_ls;
    accuracy_ls(i) = mean((pred_ls > 0.5) == b_clean);
    
    ls_test = A_test * x_ls;
    test_accuracy_ls(i) = mean((ls_test > 0.5) == b_test);
    
    %% 3.3 Train and evaluate LASSO model
    cvx_begin quiet
    variable x_lasso(m)
    minimize(sum_square(b_obs - A * x_lasso) + lambda_lasso * norm(x_lasso, 1))
    cvx_end
    pred_lasso = A * x_lasso;
    accuracy_lasso(i) = mean((pred_lasso > 0.5) == b_clean);
    
    lasso_test = A_test * x_lasso;
    test_accuracy_lasso(i) = mean((lasso_test > 0.5) == b_test);
    
    %% 3.4 Train and evaluate RLS model (C full coverage)
    % Build RLS model noise matrix C (100% coverage)
    C_full = zeros(num_samples, num_samples);
    for j = 1:num_samples
        if ismember(j, true_flip_indices)
            C_full(j, j) = 2 * b_clean(j) - 1; % D_ii = 2*b_i - 1
        end
    end
    c_diag_full = diag(C_full);
    
    cvx_begin quiet
    variable x_rls_full(m)
    minimize(sum_square(A * x_rls_full - b_obs - 0.5 * C_full * ones(num_samples,1)) + norm(C_full' * (A * x_rls_full - b_obs - 0.5 * C_full * ones(num_samples,1)), 1))
    cvx_end
    
    pred_rls_full = A * x_rls_full;
    accuracy_rls_full(i) = mean((pred_rls_full > 0.5) == b_clean);
    
    rls_full_test = A_test * x_rls_full;
    test_accuracy_rls_full(i) = mean((rls_full_test > 0.5) == b_test);
    
    %% 3.5 Train and evaluate RLS model (C partial coverage - 70%)
    % Only cover 70% of actual error labels
    num_covered = round(0.7 * num_noise);
    % Randomly select 70% of error labels for coverage
    partial_flip_indices = true_flip_indices(randperm(num_noise, num_covered));
    
    % Build RLS model noise matrix C (70% coverage)
    C_partial = zeros(num_samples, num_samples);
    for j = 1:num_samples
        if ismember(j, partial_flip_indices)
            C_partial(j, j) = 2 * b_clean(j) - 1;
        end
    end
    c_diag_partial = diag(C_partial);
    
    cvx_begin quiet
    variable x_rls_partial(m)
    minimize(sum_square(A * x_rls_partial - b_obs - 0.5 * C_partial * ones(num_samples,1)) + norm(C_partial' * (A * x_rls_partial - b_obs - 0.5 * C_partial * ones(num_samples,1)), 1))
    cvx_end
    
    pred_rls_partial = A * x_rls_partial;
    accuracy_rls_partial(i) = mean((pred_rls_partial > 0.5) == b_clean);
    
    rls_partial_test = A_test * x_rls_partial;
    test_accuracy_rls_partial(i) = mean((rls_partial_test > 0.5) == b_test);
end

LS(idx,:) = test_accuracy_ls;
LASSO(idx,:) = test_accuracy_lasso;
RLSF(idx,:) = test_accuracy_rls_full;
RLSP(idx,:) = test_accuracy_rls_partial;
end

%% 4. Plot accuracy curves vs noise ratio
acc_ls = mean(LS);
acc_lasso = mean(LASSO);
acc_rlsf = mean(RLSF);
acc_rlsp = mean(RLSP);

figure('Position', [950, 360, 510, 410]);
plot(noise_ratios, acc_ls, 'b^-', 'LineWidth', 2, 'DisplayName', 'LS');
hold on;
plot(noise_ratios, acc_lasso, 'rs-', 'LineWidth', 2, 'DisplayName', 'LASSO');
plot(noise_ratios, acc_rlsf, 'ko-', 'LineWidth', 2, 'DisplayName', 'RLS (100% C)');
plot(noise_ratios, acc_rlsp, 'g*-', 'LineWidth', 2, 'DisplayName', 'RLS (70% C)');
xlabel('Noise Ratio');
ylabel('Accuracy');
% title('Accuracy of Four Models vs Label Noise Level');
legend('Location', 'best');
grid on;
print('accuracy_comparison_plot_with_partial_C.png', '-dpng', '-r300');
fprintf('Accuracy comparison plot successfully generated and saved as accuracy_comparison_plot_with_partial_C.png\n');