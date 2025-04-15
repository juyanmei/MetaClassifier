# MicrobiomeClassifier.py 项目说明

## 项目概述
MicrobiomeClassifier.py 是一个用于处理微生物组数据的Python脚本，主要功能包括数据加载、过滤、特征选择、模型训练和评估。该脚本支持多种机器学习模型，并提供了参数调优和集成学习功能。

## 安装和运行要求

### 环境配置
```bash
# 创建Python 3.11环境
conda create -n MClassifier python=3.11

# 激活环境
conda activate MClassifier

# 安装依赖库
conda install -y scikit-learn pandas numpy matplotlib seaborn
conda install -y -c conda-forge catboost xgboost
conda install -y pytorch tqdm joblib
```

### 系统要求
- Python 3.11+
- 依赖库：
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - catboost
  - xgboost
  - torch
  - tqdm
  - joblib

## 参数说明
| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| --prof_file | str | 是 | 物种丰度数据文件路径 |
| --metadata_file | str | 是 | 元数据文件路径 |
| --target | str | 是 | 目标类型（Project/Disease） |
| --target_id | str | 否 | 目标ID |
| --models | str | 否 | 模型列表（逗号分隔），可选值：RandomForest, CatBoost, LogisticRegression, MLP, XGBoost, lasso, SVM, ElasticNet, NeuralNetwork, GaussianNB, KNN, GMWI2 （GMWI2更新中，暂不支持） |
| --feature | bool | 是 | 是否进行特征选择 |
| | | | 注意：示例数据由于数据量较小，特征选择可能存在问题，因此示例代码中都设置为No。实际数据可以正常使用特征选择功能。 |
| --turner | bool | 是 | 是否进行参数调优 |
| --ensemble | bool | 否 | 是否使用集成学习 | 更新中， 暂不支持 |
| --y_cohort_group | bool | 否 | 是否返回队列信息 |
| --y_cohort_type | str | 否 | 队列信息类型（Project/Disease），默认为Project |
| --pres | bool | 否 | 是否使用pres profile或RA profile |
| --cpu | int | 否 | CPU使用数量 |
| --repeat | int | 否 | 重复次数 |
| --cv | int | 否 | 交叉验证折数，默认为5 |
| --output | str | 是 | 输出目录 |

## 使用示例
```bash

# 基本使用（按Project分类）
# 示例1：使用特征选择的Project分类
# --prof_file: 输入的物种丰度数据文件
# --metadata_file: 样本元数据文件
# --target Project: 按项目进行分类
# --target_id ProjectA: 指定目标项目为ProjectA
# --models: 使用LASSO和逻辑回归模型（可选模型：RandomForest, CatBoost, LogisticRegression, MLP, XGBoost, lasso, SVM, ElasticNet, NeuralNetwork, GaussianNB, KNN, GMWI2）
# --feature No: 不启用特征选择
# --turner Yes: 启用参数调优
# --pres No: 使用相对丰度数据
# --repeat 10: 重复实验10次
# --cv 10: 使用10折交叉验证
# --cpu 4: 使用4个CPU核心
python MicrobiomeClassifier.py --prof_file test/prof_test.csv --metadata_file test/metadata_test.csv --target Project --target_id ProjectA --output ./result_abun --models lasso,LogisticRegression --ensemble No --feature No --turner Yes --pres No --repeat 10 --cv 10 --cpu 4

# 示例2：使用原始特征的Project分类
# 与示例1相比的主要区别：
# --feature No: 不进行特征选择
# --pres Yes: 使用presence/absence数据
python MicrobiomeClassifier.py --prof_file test/prof_test.csv --metadata_file test/metadata_test.csv --target Project --target_id ProjectA --output ./result_pres --models lasso,LogisticRegression --ensemble No --feature No --turner Yes --pres Yes --repeat 10 --cv 10 --cpu 4

# 基本使用（按Disease分类）
# 示例3：使用特征选择的Disease分类
# --target Disease: 按疾病类型分类
# --target_id DiseaseA: 指定目标疾病为DiseaseA
python MicrobiomeClassifier.py --prof_file test/prof_test.csv --metadata_file test/metadata_test.csv --target Disease --target_id DiseaseA --output ./result_abun --models lasso,LogisticRegression --ensemble No --feature No --turner Yes --pres No --repeat 10 --cv 10 --cpu 4

# 单一疾病（按Disease分类， LOCO - Leave One Cohort Out）
# 示例4：使用LOCO交叉验证
# --y_cohort_group Yes: 启用队列分组
# --y_cohort_type Project: 按项目进行队列划分
python MicrobiomeClassifier.py --prof_file test/prof_test.csv --metadata_file test/metadata_test.csv --target Disease --target_id DiseaseA --y_cohort_group Yes --y_cohort_type Project --output ./result_abun --models lasso,LogisticRegression --ensemble No --feature No --turner Yes --pres No --repeat 10 --cv 10 --cpu 4

# 全部样本（OneMajorDisease， LODO - Leave One Disease Out）
# 示例5：使用LODO交叉验证
# --target OneMajorDisease: 对主要疾病进行分类
# --y_cohort_type Disease: 按疾病类型进行队列划分
python MicrobiomeClassifier.py --prof_file test/prof_test.csv --metadata_file test/metadata_test.csv --target OneMajorDisease --y_cohort_group Yes --y_cohort_type Disease --output ./result_abun --models lasso,LogisticRegression --ensemble No --feature No --turner Yes --pres No --repeat 10 --cv 10 --cpu 4

## 输出文件说明
- scores.csv: 模型评估分数
- predictions.csv: 预测结果
- whole_predictions.csv: 完整预测结果
- features.csv: 选择的特征（如果启用特征选择）
- imp.csv: 特征重要性分数
- 各种图表文件（PDF格式）

## 注意事项
1. 确保输入文件格式正确
2. 特征选择会增加运行时间
3. 参数调优需要更多计算资源
4. 输出目录会自动创建