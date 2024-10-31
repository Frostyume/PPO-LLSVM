function [best_C, best_gamma] = pso_optimize(X_train, y_train, num_particles, max_iters, C_min, C_max, gamma_min, gamma_max, taskType)
    % 输入：
    % X_train: 训练集特征矩阵
    % y_train: 训练集标签
    % num_particles: 粒子数量
    % max_iters: 最大迭代次数
    % C_min: 正则化参数C的最小值
    % C_max: 正则化参数C的最大值
    % gamma_min: 核函数参数gamma的最小值
    % gamma_max: 核函数参数gamma的最大值
    % taskType: 任务类型（'classification' 或 'regression'）
    % 输出：
    % best_C: 优化后的正则化参数C
    % best_gamma: 优化后的核函数参数gamma
    
    % 设置随机种子以便重现结果
    rng(1);  % 你可以选择任何整数作为种子
    % 初始化粒子的位置和速度
    particles = rand(num_particles, 2) .* [C_max - C_min, gamma_max - gamma_min] + [C_min, gamma_min];
    velocities = rand(num_particles, 2) * 0.1;  % 初始化速度为小的随机值


    personal_best_positions = particles;
    personal_best_scores = inf(num_particles, 1);
    
    global_best_position = particles(1, :);
    global_best_score = inf;

    % PSO参数
    w = 0.8;   % 惯性权重
    c1 = 1.5;  % 个体加速度系数
    c2 = 1.5;  % 群体加速度系数

    for iter = 1:max_iters
        for i = 1:num_particles
            % 计算当前粒子的适应度
            C = particles(i, 1);
            gamma = particles(i, 2);
            score = lssvm_fitness(X_train, y_train, C, gamma, taskType);

            % 更新个体最佳位置和全局最佳位置
            if score < personal_best_scores(i)
                personal_best_scores(i) = score;
                personal_best_positions(i, :) = particles(i, :);
            end

            if score < global_best_score
                global_best_score = score;
                global_best_position = particles(i, :);
            end
        end

        % 更新粒子的位置和速度
        for i = 1:num_particles
            velocities(i, :) = w * velocities(i, :) ...
                + c1 * rand * (personal_best_positions(i, :) - particles(i, :)) ...
                + c2 * rand * (global_best_position - particles(i, :));

            % 限制速度
            max_velocity = [50, 50];  % 可以根据需要调整
            velocities(i, :) = max(min(velocities(i, :), max_velocity), -max_velocity);

            particles(i, :) = particles(i, :) + velocities(i, :);

            % 限制粒子的位置在参数范围内
            particles(i, 1) = min(max(particles(i, 1), C_min), C_max);
            particles(i, 2) = min(max(particles(i, 2), gamma_min), gamma_max);
        end

        % 显示当前迭代信息
        fprintf('迭代第%d次: 最佳 C = %.4f, 最佳 gamma = %.4f, 误差 = %.4f\n', ...
            iter, global_best_position(1), global_best_position(2), global_best_score);
    end

    best_C = global_best_position(1);
    best_gamma = global_best_position(2);
end
