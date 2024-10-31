function model = train_lssvm(trainX, trainY, C, gamma, taskType)
    % 输入：
    % trainX: 输入特征矩阵
    % trainY: 输出标签
    % C: 正则化参数
    % gamma: RBF核函数参数
    % taskType: 'classification' 或 'regression'

    % 样本数量
    [n, ~] = size(trainX);
    
    % 计算RBF核矩阵
    K = rbf_kernel(trainX, trainX, gamma);
    
    % 添加正则化项以避免矩阵奇异
    regularization_matrix = (1/C) * eye(n);
    Omega = [zeros(1, n+1); [ones(n, 1), K + regularization_matrix]];  % 形成Omega矩阵
    Y = [0; trainY];  % 形成Y向量

    % 求解线性方程组
    alpha_b = pinv(Omega) * Y;  % 使用伪逆计算
    model.alpha = alpha_b(2:end);  % 拉格朗日乘子
    model.b = alpha_b(1);          % 偏置项
    model.gamma = gamma;           % 核函数参数
    model.C = C;                   % 正则化参数
    model.trainX = trainX;         % 保存训练数据
    model.isClassification = strcmp(taskType, 'classification');  % 根据输入判断任务类型

    % 检查输出有效性
    if any(isnan(model.alpha)) || isinf(model.b)
        error('模型训练失败，输出参数无效。');
    end
end
