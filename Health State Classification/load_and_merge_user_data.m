function [merged_data, success] = load_and_merge_user_data(idx)
% load_and_merge_user_data Load and merge health and insole data for specified user
%
% Input:
%   idx - User number (numeric)
%
% Output:
%   merged_data - Merged table data containing all columns from both files, aligned by timestamp
%   success - Logical value indicating whether data loading and merging was successful

    % Initialize output
    merged_data = [];
    success = false;

    % Construct file paths and names
    data_folder = 'data'; % Relative path
    health_filename = sprintf('day1_%d_health_data.csv', idx);
    insole_filename = sprintf('day1_%d_insole_data.csv', idx);

    health_filepath = fullfile(data_folder, health_filename);
    insole_filepath = fullfile(data_folder, insole_filename);

    try
        % Check if files exist
        if ~exist(health_filepath, 'file')
            error('Health data file does not exist: %s', health_filepath);
        end
        if ~exist(insole_filepath, 'file')
            error('Insole data file does not exist: %s', insole_filepath);
        end

        % 1. Load data
        fprintf('Loading data for user %d...\n', idx);
        health_data = readtable(health_filepath);
        insole_data = readtable(insole_filepath);

        % 2. Data preprocessing: Assume timestamp column names are 'Timestamp' or 'time'
        %    Please adjust according to your actual file column names
        timestamp_col_health = 'time'; % Timestamp column name for health data
        timestamp_col_insole = 'time'; % Timestamp column name for insole data

        % Convert timestamp columns to datetime type
        health_data.(timestamp_col_health) = datetime(health_data.(timestamp_col_health), 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
        insole_data.(timestamp_col_insole) = datetime(insole_data.(timestamp_col_insole), 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');

        % 3. Merge using timestamp as key (inner join)
        %    This retains rows where timestamps exist in both tables
        merged_data = innerjoin(health_data, insole_data, ...
            'LeftKeys', timestamp_col_health, 'RightKeys', timestamp_col_insole); % Merge the two timestamp columns into one

        % 4. Mark operation as successful
        success = true;
        fprintf('Data for user %d loaded and merged successfully.\n', idx);

    catch ME
        % Catch and display error
        fprintf('Error loading or merging data for user %d: %s\n', idx, ME.message);
        % merged_data remains empty, success remains false
    end

end