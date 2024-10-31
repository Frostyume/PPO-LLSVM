function score = lssvm_fitness(trainX, trainY, C, gamma, taskType)
    % 输入：
    % trainX: 输入特征矩阵
    % trainY: 输出标签
    % C: 正则化参数
    % gamma: RBF核函数参数
    % taskType: 'classification' 或 'regression'
    % 训练LS-SVM并计算其在训练集上的损失

    % 训练LS-SVM
    model = train_lssvm(trainX, trainY, C, gamma, taskType);
    
    % 预测训练集的类别和概率
    [predictedY, scores] = predict_lssvm(model, trainX);

    if model.isClassification
        % 计算交叉熵损失
        numClasses = length(unique(trainY));
        trueLabels = full(sparse(1:length(trainY), trainY, 1, length(trainY), numClasses)); % One-hot编码
        score = -mean(sum(trueLabels .* log(scores + eps), 2));  % 交叉熵损失
    else
        % 计算均方误差
        score = sqrt(mean((predictedY - trainY).^2));  % 均方误差
    end
end
