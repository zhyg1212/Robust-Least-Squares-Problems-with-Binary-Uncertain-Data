%% 准确率对比图：LS, LASSO, RLS (全覆盖) 和 RLS (部分覆盖)
%  Goal: 生成四种模型在不同噪声水平下的准确率对比图。
%  用的CHEN和KELLY的内层显式解法
%  将标签相同的数据进行随机分组，分组的组数规模分别为100维和1000维
%  每个分组内的样本量相同，数据取平均。
%  数据处理步骤：
%  1.读取单个user的数据并提取 总样本量*28 的矩阵；
%  2. 将标签不是0和1的那些行去掉，这个矩阵就是作为测试是用的矩阵A，其对应的标签是测试是用的b； 3.
%  3. 统计一下当前样本还有多少行，除以100或1000并取整，得到每个分组的样本数
%  4. 按照求出的样本数，对数据进行分组无放回的抽取，并求均值，得到的 100* 28 和1000 * 28
%  的矩阵就是我们训练用的矩阵A。其对应的标签记为b。
clear; close all; clc;

r = 10000; %或3000, 矩阵A的行数, 

for idx = 1:10
[merged_data, success] = load_and_merge_user_data(idx);
noise_ratios = 0: 0.05: 0.5;
num_ratios = length(noise_ratios);

if success
    % 1. 数据名里加上用户编号
    data_name = sprintf('merged_data_user_%d', idx);
    assignin('base', data_name, merged_data); % 将数据存入工作区，名称包含用户编号
    
    % 筛选出状态为0和1的样本；
    status_vector = merged_data.status; % 假设列名为 'status'
    idx_s = find(status_vector == 1 | status_vector == 0);
    merged_data_filtered = merged_data(idx_s,:); % 过滤后的数据，用于测试
    
    columns_to_remove = {'time', 'activity', 'status', 'gait_type', 'motion_type'};
    existing_columns_to_remove = intersect(columns_to_remove, merged_data_filtered.Properties.VariableNames, 'stable');
    
    % --- 步骤2: 生成测试矩阵 A_test 和 b_test ---
    A_test =  merged_data_filtered(:, ~ismember( merged_data.Properties.VariableNames, existing_columns_to_remove));
    A_test = table2array(A_test); % 转换为数值矩阵
    b_test = merged_data_filtered.status;
    
    % --- 步骤1 & 3 & 4: 生成训练矩阵 A 和 b_clean ---
    % 提取用于训练的特征矩阵和标签
    train_features =  merged_data_filtered (:, ~ismember( merged_data.Properties.VariableNames, existing_columns_to_remove));
    train_features = table2array(train_features);
    train_labels =  merged_data_filtered .status;
    
    % 定义两个目标维度
    
    % 初始化用于存储最终训练数据的变量
    A = []; 
    b_clean = [];
    
        num_groups = r;
        total_samples = size(train_features, 1);
        
        % --- 步骤3: 计算每个分组的样本数 ---
        samples_per_group = floor(total_samples / num_groups);
        if samples_per_group == 0
            error('样本量不足以分成 %d 组。', num_groups);
        end
        
        % 实际使用的总样本数（能被组数整除）
        total_used = samples_per_group * num_groups;
        
        % --- 步骤4: 无放回抽取并求均值 ---
        % 随机打乱索引
        shuffled_idx = randperm(total_samples);
        % 取前 total_used 个索引
        used_idx = shuffled_idx(1:total_used);
        
        % 初始化该维度的训练数据
        A_dim = zeros(num_groups, size(train_features, 2));
        b_dim = zeros(num_groups, 1);
        
        for g = 1:num_groups
            % 计算当前组的索引范围
            start_idx = (g-1) * samples_per_group + 1;
            end_idx = g * samples_per_group;
            group_idx = used_idx(start_idx:end_idx);
            
            % 对组内样本求均值
            A_dim(g, :) = mean(train_features(group_idx, :), 1);
            % 对组内标签求均值（或取中位数/众数，这里用均值后四舍五入）
            b_dim(g) = round(mean(train_labels(group_idx)));
        end
        A = A_dim;
        b_clean = b_dim;
    
else
    fprintf('无法加载用户%d的数据。\n', idx);
end

[num_samples, m] = size(A); % m = 784


%% 2. 定义错误标签比例的范围
% 横坐标：错误标签的比例


% 初始化存储准确率的向量
accuracy_ls = zeros(1, num_ratios);
accuracy_lasso = zeros(1, num_ratios);
accuracy_rls_full = zeros(1, num_ratios);
accuracy_rls_partial = zeros(1, num_ratios); % 为部分覆盖的 RLS 准备

test_accuracy_ls = zeros(1, num_ratios);
test_accuracy_lasso = zeros(1, num_ratios);
test_accuracy_rls_full = zeros(1, num_ratios);
test_accuracy_rls_partial = zeros(1, num_ratios); % 为部分覆盖的 RLS 准备

%% 3. 遍历不同的噪声比例，计算所有模型的准确率
lambda_lasso = 1; % LASSO 正则化参数

for i = 1:num_ratios
    noise_ratio = noise_ratios(i);
    fprintf('Processing noise ratio = %.2f...\n', noise_ratio);
    
    %% 3.1 在标签上施加结构化噪声 (Adversarial Label Noise)
    num_noise = round(noise_ratio * num_samples);
    % 随机选择真实被翻转的样本
    true_flip_indices = randperm(num_samples, num_noise);
    
    % --- 生成带噪声的观测标签 b_obs (用于 LS 和 LASSO) ---
    b_obs = b_clean;
    b_obs(true_flip_indices) = 1 - b_obs(true_flip_indices); % 翻转标签
    
    %% 3.2 训练并评估 LS 模型
    x_ls = A \ b_obs;
    pred_ls = A * x_ls;
    accuracy_ls(i) = mean((pred_ls > 0.5) == b_clean);
    
    ls_test = A_test * x_ls;
    test_accuracy_ls(i) = mean((ls_test > 0.5) == b_test);
    
    %% 3.3 训练并评估 LASSO 模型
    cvx_begin quiet
    variable x_lasso(m)
    minimize( sum_square(b_obs - A * x_lasso) + lambda_lasso * norm(x_lasso, 1) )
    cvx_end
    pred_lasso = A * x_lasso;
    accuracy_lasso(i) = mean((pred_lasso > 0.5) == b_clean);
    
    lasso_test = A_test * x_lasso;
    test_accuracy_lasso(i) = mean((lasso_test > 0.5) == b_test);
    %% 3.4 训练并评估 RLS 模型 (C 完全覆盖)
    % 构建 RLS 模型的噪声矩阵 C (100% 覆盖)
    C_full = zeros(num_samples, num_samples);
    for j = 1:num_samples
        if ismember(j, true_flip_indices)
            C_full(j, j) = 2 * b_clean(j) - 1; % D_ii = 2*b_i - 1
        end
    end
    c_diag_full = diag(C_full);
    
    cvx_begin quiet
    variable x_rls_full(m)
    minimize( sum_square( A * x_rls_full - b_obs - 0.5 * C_full *ones(num_samples,1)) + norm(C_full' * ( A * x_rls_full - b_obs - 0.5 * C_full *ones(num_samples,1)), 1) )
    cvx_end
    
    pred_rls_full = A * x_rls_full;
    accuracy_rls_full(i) = mean((pred_rls_full > 0.5) == b_clean);
    
    rls_full_test = A_test * x_rls_full;
    test_accuracy_rls_full(i) = mean((rls_full_test > 0.5) == b_test);
    %% 3.5 训练并评估 RLS 模型 (C 部分覆盖 - 80%)
    % 只覆盖 80% 的真实错误标签
    num_covered = round(0.7 * num_noise);
    % 随机选择 80% 的错误标签进行覆盖
    partial_flip_indices = true_flip_indices(randperm(num_noise, num_covered));
    
    % 构建 RLS 模型的噪声矩阵 C (80% 覆盖)
    C_partial = zeros(num_samples, num_samples);
    for j = 1:num_samples
        if ismember(j, partial_flip_indices)
            C_partial(j, j) = 2 * b_clean(j) - 1;
        end
    end
    c_diag_partial = diag(C_partial);
    
    
    cvx_begin quiet
    variable x_rls_partial(m)
    minimize( sum_square( A * x_rls_partial - b_obs - 0.5 * C_partial *ones(num_samples,1)) + norm(C_partial' * ( A * x_rls_partial - b_obs - 0.5 * C_partial *ones(num_samples,1)), 1) )
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

%% 4. 绘制准确率随噪声比例变化的曲线
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
% title('四种模型准确率随标签噪声水平的变化');
legend('Location', 'best');
grid on;
print('accuracy_comparison_plot_with_partial_C.png', '-dpng', '-r300');
fprintf('准确率对比图已成功生成并保存为 accuracy_comparison_plot_with_partial_C.png\n');
