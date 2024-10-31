% 读取Excel数据
data = readtable('data.xlsx');

% 提取输入和输出数据 (预学习数据)
X_train = data{:, 2:7};  % 输入特征
y_train = data{:, 10};    % 输出目标
X_test = data{:, 13:18};  % 预测输入特

[processed_X_train, processed_y_train, processed_X_test] = preprocess_data(X_train, y_train, X_test);

% 参数设置
num_particles = 5;
max_iters = 50;
C_min = 0.0001;
C_max = 10000;
gamma_min = 0.001;
gamma_max = 0.1;
taskType = 'regression';  % 任务类型

% 使用PSO优化LS-SVM的超参数 C 和 gamma
[best_C, best_gamma] = pso_optimize(processed_X_train, processed_y_train, num_particles, max_iters, C_min, C_max, gamma_min, gamma_max, taskType);

% 训练LS-SVM模型
model = train_lssvm(processed_X_train, processed_y_train, best_C, best_gamma, taskType);

% 预测新数据
y_pred = predict_lssvm(model, processed_X_train);

% 显示预测结果
disp('预测结果:');
disp(y_pred);
