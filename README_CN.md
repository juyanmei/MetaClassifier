# metaClassifier v1.0

**宏基因组分类模型构建框架** - 基于两阶段架构的完整机器学习流水线

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**语言**: [English](README.md) | [中文](README_CN.md)

## 📋 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [安装](#安装)
- [使用指南](#使用指南)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 概述

metaClassifier v1.0 是一个专为宏基因组数据设计的机器学习分类框架，采用**两阶段架构**实现无偏性能估计和稳定特征选择：

1. **第一阶段：嵌套CV评估** - 无偏性能估计 + 共识特征选择
2. **第二阶段：最终模型训练** - 使用共识特征集 + 超参数调优

### 设计理念

- ✅ **无偏性能估计**：严格的嵌套交叉验证确保性能评估的可靠性
- ✅ **稳定特征选择**：通过内层CV的共识机制筛选稳定特征
- ✅ **宏基因组优化**：针对宏基因组数据特点的预处理和特征工程
- ✅ **完整可重现性**：详细记录所有实验参数和结果

## 核心特性

### 🎯 两阶段架构

**第一阶段：嵌套CV评估**
- 嵌套交叉验证（支持Repeated K-Fold和LOCO策略）
- 联合特征选择和超参数调优
- 共识特征集生成
- 无偏性能指标计算

**第二阶段：最终模型训练**
- 基于共识特征集训练最终模型
- 独立的超参数调优
- 模型保存和部署准备

### 🔬 宏基因组特定功能

- **自适应方差过滤**：根据p/n比动态调整过滤强度
- **CLR变换**：处理组成型数据的中心对数比变换
- **有无数据支持**：支持相对丰度和有无（presence/absence）数据
- **队列分析**：支持Leave-One-Cohort-Out (LOCO)交叉验证

### 🤖 模型支持

支持多种机器学习模型：
- **LASSO** - 线性模型，特征选择能力强
- **Elastic Net** - 结合L1和L2正则化
- **Logistic Regression** - 经典逻辑回归
- **Random Forest** - 集成树模型
- **CatBoost** - 梯度提升树

### 📊 报告生成系统

支持多种分析场景的报告生成：
- `within_disease` - 疾病内项目间比较
- `between_project` - 项目间交叉验证
- `between_disease` - 疾病间交叉验证
- `overall` - 整体性能分析
- `models` - 多模型比较
- `predict_external_disease` - 外部疾病预测
- `predict_external_overall` - 外部整体预测

### 🎨 可视化功能

- ROC曲线绘制（支持repeat均值ROC）
- 性能指标热图
- 箱线图比较
- 特征重要性可视化

## 快速开始

### 基本使用

#### 1. 构建模型（build命令）

```bash
metaClassifier build \
    --prof_file data/profile.csv \
    --metadata_file data/metadata.csv \
    --model_name lasso \
    --outer_cv_folds 5 \
    --inner_cv_folds 3 \
    --outer_cv_repeats 1 \
    --output results/
```

#### 2. 生成报告（report命令）

```bash
# 疾病内项目间比较
metaClassifier report \
    --scenario within_disease \
    --metadata_file data/metadata.csv \
    --models lasso,catboost \
    --metric auc \
    --output results/

# 项目间交叉验证
metaClassifier report \
    --scenario between_project \
    --metadata_file data/metadata.csv \
    --models lasso \
    --metric auc \
    --output results/
```

## 安装

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/juyanmei/MetaClassifier.git
cd MetaClassifier

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 依赖要求

**核心依赖：**
- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

**可选依赖（用于高级功能）：**
- xgboost >= 1.5.0
- catboost >= 1.0.0
- optuna >= 3.0.0（用于贝叶斯优化）

## 使用指南

### Build命令参数

```bash
metaClassifier build [OPTIONS]

必需参数：
  --prof_file PATH           Profile数据文件路径（行=样本，列=物种）
  --metadata_file PATH       元数据文件路径

模型参数：
  --model_name {lasso,elasticnet,logistic,randomforest,catboost,neuralnetwork}
                            模型名称（默认：lasso）

交叉验证参数：
  --outer_cv_strategy {kfold,loco}
                            外层CV策略（默认：kfold）
  --outer_cv_folds INT      外层CV折数（默认：5）
  --inner_cv_folds INT      内层CV折数（默认：3）
  --outer_cv_repeats INT    外层CV重复次数（默认：1）

数据处理参数：
  --use_presence_absence    使用有无数据（默认：True）
  --use_clr                 应用CLR变换（默认：False）
  --enable_adaptive_filtering
                            启用自适应方差过滤（默认：True）

特征选择参数：
  --feature_selection       启用特征选择（默认：True）
  --feature_threshold FLOAT 一致特征频率阈值（默认：0.5）

超参数调优参数：
  --search_method {grid,random,bayes}
                            超参数搜索方法（默认：grid）
  --final_cv_folds INT      最终模型阶段CV折数（默认：5）
  --final_search_method {grid,random,bayes}
                            最终模型阶段搜索方法

输出参数：
  --output PATH             结果输出目录
  --cpu INT                 CPU核心数（默认：4）
```

### Report命令参数

```bash
metaClassifier report [OPTIONS]

必需参数：
  --scenario {within_disease,between_project,between_disease,overall,models,predict_external_disease,predict_external_overall}
                            分析场景
  --metadata_file PATH      元数据文件路径

可选参数：
  --models MODEL_LIST       模型列表（逗号分隔）
  --metric {auc,accuracy}   评估指标（默认：auc）
  --output PATH             结果输出目录
  --builds_root PATH        构建结果根目录
  --emit_predictions        生成预测结果
```

## 输出结果

### Build命令输出结构

```
output/
├── 1_performance_metrics/        # 性能指标
│   ├── nested_cv_pred_proba.csv # OOF预测概率
│   ├── nested_cv_summary.csv     # 性能汇总
│   └── ...
├── 2_final_model/                # 最终模型
│   ├── consensus_features.json   # 共识特征集
│   ├── final_training_results.json
│   └── ...
├── 3_hyperparameter_analysis/    # 超参数分析
└── 4_reproducibility/            # 可重现性信息
    ├── run.log                   # 运行日志
    └── final_run.yaml            # 完整配置
```

### Report命令输出结构

```
output/reports/
├── within_disease/               # 疾病内分析
├── between_project/              # 项目间分析
├── between_disease/               # 疾病间分析
├── overall/                      # 整体分析
├── models/                       # 模型比较
└── predict_external_*/          # 外部预测
```

## 贡献指南

我们欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black src/

# 类型检查
mypy src/
```

## 常见问题

### Q: 如何选择合适的CV策略？

**A:** 
- **K-Fold**：适用于样本量较大、队列分布均匀的情况
- **LOCO (Leave-One-Cohort-Out)**：适用于需要评估跨队列泛化能力的情况
- **Repeated K-Fold**：需要更稳定的性能估计时使用

### Q: 如何解释共识特征？

**A:** 共识特征是在内层CV的多个折中稳定出现的特征，表示这些特征对模型性能有稳定的贡献。

### Q: AUC计算为什么重要？

**A:** 正确的AUC计算方式（按repeat聚合）能够更准确地反映模型的真实性能，避免因每折样本量不同导致的偏差。

## 更新日志

### v1.0.0 (当前版本)

**重大更新：**
- ✅ 修复AUC计算方式：从每折AUC均值改为按repeat计算整体AUC
- ✅ 完善报告生成系统：支持多种分析场景
- ✅ 优化可视化功能：支持repeat均值ROC曲线
- ✅ 增强可重现性：详细记录所有实验参数

**新功能：**
- 支持Repeated K-Fold交叉验证
- 扩展的模型支持（CatBoost, XGBoost等）
- 完整的报告生成流水线
- 自适应方差过滤优化

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

感谢所有为本项目做出贡献的开发者和研究者。

## 联系方式

- **Issues**: [GitHub Issues](https://github.com/juyanmei/MetaClassifier/issues)
- **文档**: [GitHub仓库](https://github.com/juyanmei/MetaClassifier)

---

**metaClassifier v1.0** - 让宏基因组分类更简单、更可靠、更可重现 🧬🔬

