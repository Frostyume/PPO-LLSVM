function [processed_X_train, processed_y_train, processed_X_test] = preprocess_data(X_train, y_train, X_test)
    % 填充缺失值
    processed_X_train = fillmissing(X_train, 'constant', mean(X_train, 'omitnan'));
    processed_y_train = fillmissing(y_train, 'constant', mean(y_train, 'omitnan'));
    
    % 打乱数据
    num_samples = size(processed_X_train, 1);
    idx = randperm(num_samples);
    processed_X_train = processed_X_train(idx, :);
    processed_y_train = processed_y_train(idx);
    
    % 数据归一化（Min-Max 归一化）
    X_min = min(processed_X_train);
    X_max = max(processed_X_train);
    processed_X_train = (processed_X_train - X_min) ./ (X_max - X_min);
    
    % 提取并处理预测数据
    processed_X_test = fillmissing(X_test, 'constant', mean(X_test, 'omitnan'));
    
    % 使用训练数据的最小值和最大值进行归一化
    processed_X_test = (processed_X_test - X_min) ./ (X_max - X_min);
    
    % 数据标准化
    processed_X_train = (processed_X_train - mean(processed_X_train)) ./ std(processed_X_train, 0, 1);
    processed_X_test = (processed_X_test - mean(processed_X_train)) ./ std(processed_X_train, 0, 1);  % 使用训练集均值和标准差
    
    % 检查标准化后的数据是否存在NaN
    processed_X_train = fillmissing(processed_X_train, 'constant', 0);
    processed_X_test = fillmissing(processed_X_test, 'constant', 0);
end
