function [predicted, scores] = predict_lssvm(model, testX)
    % 输入：
    % model: 训练好的LS-SVM模型
    % testX: 待预测的输入数据
    % 输出：
    % predicted: 预测的输出结果（类别标签或回归值）
    % scores: 每个类别的概率分布（仅在分类任务中）

    % 计算RBF核矩阵
    K = rbf_kernel(testX, model.trainX, model.gamma);
    
    % 计算决策值
    decisionValues = K * model.alpha + model.b;

    if model.isClassification
        % 通过Softmax将决策值转换为概率
        scores = softmax(decisionValues);
        % 返回预测的类别标签
        [~, predicted] = max(scores, [], 2);  % 选择概率最大的类别
    else
        % 回归任务直接返回决策值
        predicted = decisionValues;  % 直接返回预测值
        scores = [];  % 对于回归，不需要概率分布
    end
end

function p = softmax(z)
    % 计算Softmax概率分布
    expZ = exp(z - max(z, [], 2));  % 数值稳定性处理
    p = expZ ./ sum(expZ, 2);
end
