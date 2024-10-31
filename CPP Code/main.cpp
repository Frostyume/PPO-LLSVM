#include "liblssvm.h"  // MATLAB生成的库头文件
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <tuple>
using namespace std;

vector<double> calculateColumnMeans(const vector<vector<double>>& data) {
    if (data.empty() || data[0].empty()) return {}; // 检查数据是否为空

    size_t numCols = data[0].size();
    vector<double> means(numCols, 0.0);
    vector<int> counts(numCols, 0);

    for (const auto& row : data) {
        for (size_t j = 0; j < numCols; ++j) {
            if (!isnan(row[j])) {
                means[j] += row[j];
                counts[j]++;
            }
        }
    }

    for (size_t j = 0; j < numCols; ++j) {
        if (counts[j] > 0) {
            means[j] /= counts[j]; // 计算列均值
        }
        else {
            means[j] = NAN; // 如果该列全是 NaN，设置为 NaN
        }
    }

    return means;
}

// 用列均值替换 NaN
void fillmissing(vector<vector<double>>& data) {
    auto means = calculateColumnMeans(data);

    for (auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            if (isnan(row[j])) {
                row[j] = means[j]; // 用该列的均值替换 NaN
            }
        }
    }
}

// 将二维 vector 转换为 mwArray 类型
mwArray convertToMwArray(vector<vector<double>>& input) {
    // 获取行数和列数
    size_t rows = input.size();
    size_t cols = input[0].size();
    fillmissing(input);

    // 创建mwArray（行主序）
    mwArray array(rows, cols, mxDOUBLE_CLASS);

    // 将数据线性化并填充(列主序)
    vector<double> flatData(rows * cols);
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            flatData[j * rows + i] = input[i][j]; // 按列填充
        }
    }

    // 打印flatData
    //cout << "flatData: ";
    //for (const auto& value : flatData) {
    //    cout << value << " ";
    //}
    //cout << endl;

    // 使用SetData方法填充数据
    array.SetData(flatData.data(), rows * cols);
    return array;
}


// 初始化模型参数
void model_init(mwArray& model, const mwArray& trainX) {
    mwArray alpha_data(trainX.NumberOfDimensions(), 1, mxDOUBLE_CLASS);
    mwArray b_data(1, 1, mxDOUBLE_CLASS);
    mwArray gamma_data(1, 1, mxDOUBLE_CLASS);
    mwArray C_data(1, 1, mxDOUBLE_CLASS);
    mwArray isClassification_data(1, 1, mxINT32_CLASS);

    double alpha[1] = { 0.0 }; // 初始化 alpha
    double b[1] = { 0.0 }; // 初始化 b
    double gamma[1] = { 0.0 }; // 初始化 gamma
    double C[1] = { 0.0 }; // 初始化 C
    int isClassification[1] = { 0 }; // 初始化分类标志

    // 设置数据
    alpha_data.SetData(alpha, 1);
    b_data.SetData(b, 1);
    gamma_data.SetData(gamma, 1);
    C_data.SetData(C, 1);
    isClassification_data.SetData(isClassification, 1);

    // 将参数赋值给模型
    model(1) = alpha_data;
    model(2) = b_data;
    model(3) = gamma_data;
    model(4) = C_data;
    model(5) = trainX;
    model(6) = isClassification_data;

    cout << "model初始化完成" << endl;
}

// LSSVM 类定义
class LSSVM {
public:
    // 拟合模型
    tuple<mwArray, mwArray, mwArray>process(vector<vector<double>>& trainX, vector<double>& trainY, vector<vector<double>>& testX) {
        mwArray X_train(trainX.size(), trainX[0].size(), mxDOUBLE_CLASS);
        mwArray y_train(trainY.size(), 1, mxDOUBLE_CLASS);
        mwArray X_test(testX.size(), testX[0].size(), mxDOUBLE_CLASS);
        mwArray processed_X_train(trainX.size(), trainX[0].size(), mxDOUBLE_CLASS);
        mwArray processed_y_train(trainY.size(), 1, mxDOUBLE_CLASS);
        mwArray processed_X_test(testX.size(), testX[0].size(), mxDOUBLE_CLASS);
        X_train = convertToMwArray(trainX);
        y_train.SetData(trainY.data(), trainY.size());
        X_test = convertToMwArray(testX);
        preprocess_data(3, processed_X_train, processed_y_train, processed_X_test, X_train, y_train, X_test);
        return make_tuple(processed_X_train, processed_y_train, processed_X_test);
    }

    mwArray fit(const mwArray& X_train, const mwArray& y_train,
        int num, int iters, double Cmin, double Cmax, double gam_min, double gam_max, const char* task_type) {
        const char* fields[] = { "alpha","b","gamma", "C", "trainX", "isClassification"};
        mwArray model(1, 1, 6, fields);

        model_init(model, X_train);

        // 检查数据创建是否成功
        if (!X_train) {
            cerr << "创建 X_train 失败!" << endl;
            return {};
        }
        if (!y_train) {
            cerr << "创建 y_train 失败!" << endl;
            return {};
        }

        // 调用PSO优化函数
        mwArray best_C(1, 1, mxDOUBLE_CLASS);
        mwArray best_gamma(1, 1, mxDOUBLE_CLASS);
        mwArray num_particles(1, 1, mxDOUBLE_CLASS);
        mwArray max_iters(1, 1, mxDOUBLE_CLASS);
        mwArray C_min(1, 1, mxDOUBLE_CLASS);
        mwArray C_max(1, 1, mxDOUBLE_CLASS);
        mwArray gamma_min(1, 1, mxDOUBLE_CLASS);
        mwArray gamma_max(1, 1, mxDOUBLE_CLASS);
        mwArray taskType(task_type);

        // 设置参数
        num_particles.SetData(&num, 1);
        max_iters.SetData(&iters, 1);
        C_min.SetData(&Cmin, 1);
        C_max.SetData(&Cmax, 1);
        gamma_min.SetData(&gam_min, 1);
        gamma_max.SetData(&gam_max, 1);

        cout << "准备调用 pso_optimize" << endl;
        // 调用pso_optimize函数
        try {
            pso_optimize(2, best_C, best_gamma, X_train, y_train, num_particles, max_iters, C_min, C_max, gamma_min, gamma_max, taskType);
        cout << "调用 pso_optimize 完成" << endl;
        cout << "Best C: " << best_C << ", Best gamma: " << best_gamma << endl;
        }
        catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
        }

        // 调用train_lssvm函数训练模型
        cout << "准备调用 train_lssvm" << endl;
        try {
            train_lssvm(1, model, X_train, y_train, best_C, best_gamma, taskType);
            cout << "调用 train_lssvm 完成" << endl;
            cout << "训练完成" << endl;
        }
        catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
        }
        return model;
    }

    // 预测函数
    vector<double> predict(const mwArray& X_test, mwArray& model) {
        int rows = X_test.GetDimensions()(1);
        mwArray predicted(rows, 1, mxDOUBLE_CLASS);
        mwArray scores(rows, 1, mxDOUBLE_CLASS);
        // 调用 predict_lssvm函数
        cout << "准备调用 predict_lssvm" << endl;
        try {
            predict_lssvm(2, predicted, scores, model, X_test);
            cout << "调用 predict_lssvm 完成" << endl;
        }
        catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
        }
        if (!predicted) {
            cerr << "获取预测结果失败!" << endl;
            return {};
        }

        // 转换结果到 vector
        vector<double> result(rows);
        predicted.GetData(result.data(), rows);

        return result;
    }
};

int main() {
    if (!mclInitializeApplication(NULL, 0) || !liblssvmInitialize()){
        cerr << "初始化失败!" << endl;
        return 0;
    }
    srand(static_cast<unsigned int>(time(0))); // 用于随机数生成

    // 预学习数据
    vector<vector<double>> trainingInput = {
 {43.20000076, 34.90000153, 34.04000092, 43.20000076, 5.960000038, 4.019999981},
        {43.70000076, 34.40000153, 33.65999985, 42.29999924, 0.029999999, 1.370000005},
        {43.90000153, 34.79999924, 33.24000168, 41.40000153, 0.059999999, 2.140000105},
        {44.09999847, 34.70000076, 33.09000015, 40.59999847, 0, 1.980000019},
        {44.29999924, 34.90000153, 32.66999817, 39.79999924, 0, 1.580000043},
        {44.40000153, 35.20000076, 32.49000168, 39.20000076, 0, 1.870000005},
        {44.29999924, 35.70000076, 32.29000092, 38.59999847, 0, 2.74000001},
        {44.20000076, 35.40000153, 32.27000046, NAN, 0.029999999, 2.039999962},
        {44.09999847, 35.59999847, 32.09000015, 37.79999924, 0.029999999, 1.059999943},
        {44, 35.70000076, 0, 37.5, 0.029999999, 1.799999952},
        {43.90000153, 36.09999847, 31.92000008, 37.40000153, 6.199999809, 5.130000114},
        {43.70000076, 36.79999924, 31.88999939, 38, 6.210000038, 3.109999895},
        {43.59999847, 37.20000076, 31.88999939, 38.40000153, 6.190000057, 3.279999971},
        {43.59999847, 37.40000153, 32.09000015, 38.90000153, 6.190000057, 3.5},
        {43.70000076, 37.70000076, 32.09000015, 39.29999924, 6.21999979, 4.46999979},
        {43.79999924, 38.09999847, 32.06999969, 39.79999924, NAN, 3.390000105},
        {43.90000153, 38.20000076, 31.92000008, 40.20000076, 6.179999828, 4.769999981},
        {44, 38.40000153, 31.92000008, 40.59999847, 6.179999828, 3.210000038},
        {44.29999924, 38.20000076, 31.71999931, 40.59999847, 0.119999997, 2.349999905},
        {44.40000153, 38, 31.52000046, 40.59999847, 6.21999979, 2.529999971},
        {44.5, 37.29999924, 31.48999977, 40.79999924, 6.21999979, 3.75},
        {44.70000076, NAN, 31.31999969, 40.90000153, 6.210000038, 5.28000021},
        {44.70000076, 36.09999847, 31.35000038, 0, 6.21999979, 4.559999943},
        {44.90000153, 35.90000153, 31.14999962, 41.20000076, 6.190000057, 3.569999933},
        {45.09999847, 35.79999924, 30.95000076, 41.40000153, 6.190000057, 4.800000191},
        {45.29999924, 35.79999924, 30.77000046, 41.40000153, 6.21999979, 5},
        {45.5, 35.79999924, 30.77000046, 41.59999847, 6.239999771, 4.989999771},
        {45.59999847, 0, 30.54999924, 41.70000076, 6.179999828, 5.239999771},
        {45.90000153, 36.20000076, 30.35000038, 41.70000076, 6.199999809, 6},
        {46.09999847, 36.40000153, 30.17000008, 41.79999924, 6.880000114, 7.5},
        {46.40000153, 36.79999924, 30.17000008, 42, 6.929999828, 7.46999979},
        {46.79999924, 37.09999847, 29.97999954, 42.20000076, 7.130000114, 7.769999981},
        {47.20000076, 37.40000153, 29.60000038, 42.40000153, 7.690000057, 8.140000343},
        {47.70000076, 37.5, 29.20000076, 42.70000076, 7.820000172, 7.829999924},
        {48.29999924, 37.70000076, 28.97999954, 43, 8.720000267, 7.980000019},
        {48.79999924, 38.20000076, 28.43000031, 43.20000076, 7.96999979, 7.059999943},
        {49.59999847, 38.59999847, 28.05999947, 43.70000076, 9.109999657, 8.020000458},
        {50.40000153, 39, 27.45999908, 44, 8.43999958, 8.350000381},
        {51.20000076, 39.70000076, 26.70999908, 44.40000153, 9.239999771, 8.760000229}
    };

    vector<double> trainingOutput = {
 51.70000076,
        51.40000153,
        51.09999847,
        50.70000076,
        50.40000153,
        49.90000153,
        49.5,
        48.79999924,
        48.5,
        48,
        47.70000076,
        47.59999847,
        47.70000076,
        47.90000153,
        48.20000076,
        48.59999847,
        48.79999924,
        49,
        49.29999924,
        49.20000076,
        49.20000076,
        49.20000076,
        49.29999924,
        49.5,
        49.59999847,
        49.59999847,
        49.70000076,
        49.79999924,
        49.90000153,
        49.90000153,
        50,
        50.09999847,
        50.29999924,
        50.40000153,
        50.5,
        50.79999924,
        51.09999847,
        51.29999924,
        51.70000076
    };

    // 测试数据
    vector<vector<double>> testInput = {
       {52.20000076, 40.59999847, 26.11000061, 44.70000076, 9.350000381, 9.020000458},
       {53.29999924, 41.59999847, 25.54000092, 45.09999847, 9.18999958, 9.800000191},
       {54.5, 42.59999847, 24.96999931, 45.29999924, 9.010000229, 8.739999771},
       {55.5, 43.40000153, 24.02000046, 45.59999847, 9.770000458, 10.48999977},
       {56.59999847, 43.90000153, 23.42000008, 45.79999924, 8.989999771, 10.35000038},
       {57.70000076, 44.40000153, 22.85000038, 45.90000153, 8.649999619, 8.199999809},
       {58.59999847, 45, 22.10000038, 46.20000076, 9.270000458, 8.81000042},
       {56.70000076, 44.70000076, 22.25, 46.59999847, NAN, 9.489999771},
       {54.70000076, 44.40000153, 22.87000084, 47.5, 7.659999847, 8.130000114},
       {51.5, 43, 24.77000046, 48.29999924, 7.710000038, 8.409999847},
       {49.70000076, 41.70000076, 26.68000031, 48.70000076, 7.570000172, 9.199999809},
       {47.20000076, 40.59999847, 29.20000076, 48.90000153, 6.860000134, 5.78000021},
       {48.20000076, 39.40000153, 0.0, 48.90000153, 6.53000021, 5.980000019},
       {48.70000076, 39, 29.97999954, 48.70000076, 7.820000172, 8.399999619},
       {48.70000076, 39, 29.57999992, 48.09999847, 9.180000305, 8.760000229},
       {48.90000153, 39, 29.39999962, 47.59999847, 7.860000134, 8.909999847},
       {49.59999847, 39.20000076, 28.82999992, 47.59999847, 9.880000114, 12.23999977},
       {50.70000076, 40.09999847, 28.22999954, 47.59999847, 9.93999958, 13.69999981},
       {52.79999924, 42.20000076, 27.09000015, 47.79999924, 9.25, 6.139999866},
       {55.29999924, 44.70000076, 25.94000053, 47.90000153, 9.859999657, 10.52000046},
       {55.59999847, 44.70000076, 23.62000084, 48.40000153, 10.17000008, 11.88000011},
       {55.29999924, 45.40000153, 22.87000084, 49.40000153, 10.09000015, 13.73999977},
       {55.70000076, 46.20000076, 22.87000084, 50, 10.18999958, 15.78999996},
       {56.40000153, 47.29999924, NAN, 50.5, 9.829999924, 7.409999847},
       {55.90000153, 48, 22.46999931, 51.09999847, 0.0, 12.65999985},
       {55.79999924, 48.20000076, 22.62000084, 51.5, 9.93999958, 6.809999943},
       {55.29999924, 48.40000153, 22.85000038, 52, 9.880000114, 13.39999962},
       {55.29999924, 49.09999847, 23.25, 52.09999847, 10.22000027, 14.02999973},
       {56.29999924, 49.59999847, 23.04999924, 52.20000076, 10.07999992, 14.23999977},
       {60.5, 49.20000076, 21.70000076, 52.40000153, 9.890000343, 13.38000011},
       {60.09999847, 48.79999924, 21.28000069, 52.79999924, 9.819999695, 14.21000004},
       {59.90000153, 0.0, 20.90999985, 53.09999847, 10.01000023, 12.44999981},
       {59.79999924, 48.79999924, 20.72999954, 53.5, 9.850000381, 10.14000034},
       {58.5, 48.20000076, 21.10000038, 53.79999924, 8.789999962, 8.909999847},
       {55.90000153, 47.29999924, 21.87999916, 54.20000076, 9.399999619, 9.430000305},
       {53.40000153, 45.90000153, 23.25, 54.5, 9.989999771, 10.81000042},
       {51.59999847, 45, 24.59000015, 54.5, 9.699999809, 10.97000027},
       {46.70000076, 43.5, 27.05999947, 54.5, 9.920000076, 10.63000011},
       {45, 41.90000153, 31.52000046, 54.29999924, 9.979999542, 12.06999969}
    };


    // 设置参数
    int num_particles = 5;
    int max_iters = 50;
    double C_min = 0.001;
    double C_max = 10000;
    double gamma_min = 0.001;
    double gamma_max = 0.1;
    const char* taskType = "regression"; // taskType: "classification" 或 "regression"

    // 创建 LSSVM 对象并训练模型
    LSSVM lssvm;

    const char* fields[] = { "alpha","b","gamma", "C", "trainX", "isClassification" };
    mwArray model(1, 1, 6, fields);
    mwArray X_train(trainingInput.size(), trainingInput[0].size(), mxDOUBLE_CLASS);
    mwArray y_train(trainingOutput.size(), 1, mxDOUBLE_CLASS);
    mwArray X_test(testInput.size(), testInput[0].size(), mxDOUBLE_CLASS);
    auto processed_data = lssvm.process(trainingInput, trainingOutput, trainingInput);
    X_train = get<0>(processed_data);
    y_train = get<1>(processed_data);
    X_test = get<2>(processed_data);

    model_init(model, X_train);
    model = lssvm.fit(X_train, y_train, num_particles, max_iters, C_min, C_max, gamma_min, gamma_max, taskType);
  
    // 预测
    vector<double> predictions = lssvm.predict(X_train, model);
    cout << "预测结果:" << endl;
    for (const auto& pred : predictions) {
        cout << pred << endl;
    }

    liblssvmTerminate();
    mclTerminateApplication();
    return 0;
}
