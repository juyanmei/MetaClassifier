# metaClassifier v1.0

**Microbiome Classification Framework** - A complete machine learning pipeline based on two-stage architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Language**: [English](README.md) | [中文](README_CN.md)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

metaClassifier v1.0 is a machine learning classification framework designed specifically for microbiome data, implementing **two-stage architecture** for unbiased performance estimation and stable feature selection:

1. **Stage 1: Nested CV Evaluation** - Unbiased performance estimation + Consensus feature selection
2. **Stage 2: Final Model Training** - Using consensus feature set + Hyperparameter tuning

### Design Philosophy

- ✅ **Unbiased Performance Estimation**: Strict nested cross-validation ensures reliable performance assessment
- ✅ **Stable Feature Selection**: Consensus mechanism through inner CV to select stable features
- ✅ **Microbiome-Optimized**: Preprocessing and feature engineering tailored for microbiome data characteristics
- ✅ **Complete Reproducibility**: Detailed recording of all experimental parameters and results

## Key Features

### 🎯 Two-Stage Architecture

**Stage 1: Nested CV Evaluation**
- Nested cross-validation (supports Repeated K-Fold and LOCO strategies)
- Joint feature selection and hyperparameter tuning
- Consensus feature set generation
- Unbiased performance metrics calculation

**Stage 2: Final Model Training**
- Train final model based on consensus feature set
- Independent hyperparameter tuning
- Model saving and deployment preparation

### 🔬 Microbiome-Specific Features

- **Adaptive Variance Filtering**: Dynamically adjusts filtering intensity based on p/n ratio
- **CLR Transformation**: Centered log-ratio transformation for compositional data
- **Presence/Absence Support**: Supports both relative abundance and presence/absence data
- **Cohort Analysis**: Supports Leave-One-Cohort-Out (LOCO) cross-validation

### 🤖 Model Support

Supports multiple machine learning models:
- **LASSO** - Linear model with strong feature selection capability
- **Elastic Net** - Combines L1 and L2 regularization
- **Logistic Regression** - Classic logistic regression
- **Random Forest** - Ensemble tree model
- **CatBoost** - Gradient boosting tree
- **XGBoost** - Extreme gradient boosting
- **SVM** - Support Vector Machine
- **Neural Network** - Neural network
- **KNN** - K-Nearest Neighbors
- **Gaussian Naive Bayes** - Gaussian Naive Bayes

### 📊 Reporting System

Supports multiple analysis scenarios:
- `within_disease` - Within-disease inter-project comparison
- `between_project` - Inter-project cross-validation
- `between_disease` - Inter-disease cross-validation
- `overall` - Overall performance analysis
- `models` - Multi-model comparison
- `predict_external_disease` - External disease prediction
- `predict_external_overall` - External overall prediction

### 🎨 Visualization Features

- ROC curve plotting (supports repeat-mean ROC)
- Performance metrics heatmaps
- Boxplot comparisons
- Feature importance visualization

## Architecture

### Two-Stage Workflow

```
Data Loading & Preprocessing
    ↓
┌─────────────────────────────────────┐
│  Stage 1: Nested CV Evaluation     │
│  - Outer CV loop                    │
│  - Inner CV: Feature selection +    │
│    Hyperparameter tuning            │
│  - Consensus feature set generation │
│  - Performance metrics calculation  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Stage 2: Final Model Training      │
│  - Use consensus feature set        │
│  - Hyperparameter tuning            │
│  - Model training & saving          │
└─────────────────────────────────────┘
    ↓
Model Deployment & Report Generation
```

## Quick Start

### Basic Usage

#### 1. Build Model (build command)

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

#### 2. Generate Report (report command)

```bash
# Within-disease inter-project comparison
metaClassifier report \
    --scenario within_disease \
    --metadata_file data/metadata.csv \
    --models lasso,catboost \
    --metric auc \
    --output results/

# Inter-project cross-validation
metaClassifier report \
    --scenario between_project \
    --metadata_file data/metadata.csv \
    --models lasso \
    --metric auc \
    --output results/
```

## Installation

### Install from Source

```bash
# Clone repository
git clone https://github.com/juyanmei/MetaClassifier.git
cd MetaClassifier

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

**Core Dependencies:**
- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

**Optional Dependencies (for advanced features):**
- xgboost >= 1.5.0
- catboost >= 1.0.0
- optuna >= 3.0.0 (for Bayesian optimization)

## Usage Guide

### Build Command Parameters

```bash
metaClassifier build [OPTIONS]

Required Parameters:
  --prof_file PATH          Profile data file path (rows=samples, cols=species)
  --metadata_file PATH      Metadata file path

Model Parameters:
  --model_name {lasso,elasticnet,logistic,randomforest,catboost,neuralnetwork}
                            Model name (default: lasso)

Cross-Validation Parameters:
  --outer_cv_strategy {kfold,loco}
                            Outer CV strategy (default: kfold)
  --outer_cv_folds INT      Number of outer CV folds (default: 5)
  --inner_cv_folds INT      Number of inner CV folds (default: 3)
  --outer_cv_repeats INT    Number of outer CV repeats (default: 1)

Data Processing Parameters:
  --use_presence_absence    Use presence/absence data (default: True)
  --use_clr                 Apply CLR transformation (default: False)
  --enable_adaptive_filtering
                            Enable adaptive variance filtering (default: True)

Feature Selection Parameters:
  --feature_selection       Enable feature selection (default: True)
  --feature_threshold FLOAT Consensus feature frequency threshold (default: 0.5)

Hyperparameter Tuning Parameters:
  --search_method {grid,random,bayes}
                            Hyperparameter search method (default: grid)
  --final_cv_folds INT      CV folds for final model stage (default: 5)
  --final_search_method {grid,random,bayes}
                            Search method for final model stage

Output Parameters:
  --output PATH             Output directory
  --cpu INT                 Number of CPU cores (default: 4)
```

### Report Command Parameters

```bash
metaClassifier report [OPTIONS]

Required Parameters:
  --scenario {within_disease,between_project,between_disease,overall,models,predict_external_disease,predict_external_overall}
                            Analysis scenario
  --metadata_file PATH      Metadata file path

Optional Parameters:
  --models MODEL_LIST       Model list (comma-separated)
  --metric {auc,accuracy}    Evaluation metric (default: auc)
  --output PATH             Output directory
  --builds_root PATH        Build results root directory
  --emit_predictions        Generate prediction results
```

## Project Structure

```
metaClassifier_v1.0/
├── src/
│   └── metaClassifier/          # Main source code
│       ├── cli/                  # Command-line interface
│       ├── core/                 # Core functionality modules
│       ├── data/                 # Data processing
│       ├── models/               # Model implementations
│       ├── pipelines/            # Pipelines
│       ├── evaluation/           # Evaluation modules
│       ├── preprocessing/        # Preprocessing modules
│       ├── config/              # Configuration files
│       ├── utils/               # Utility functions
│       ├── extended/            # Extended features
│       └── main.py              # Main entry point
├── docs/                        # Documentation
│   ├── nested_cv_architecture.md
│   ├── inner_cv_logic_detailed.md
│   └── ...
├── tests/                       # Test suite
├── requirements.txt             # Dependencies
├── setup.py                    # Installation script
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Documentation

Detailed documentation is located in the `docs/` directory:

- [Nested CV Architecture](docs/nested_cv_architecture.md) - Detailed design of nested cross-validation
- [Inner CV Logic Detailed](docs/inner_cv_logic_detailed.md) - Joint optimization mechanism of inner CV
- [Joint Optimization YAML Info](docs/joint_optimization_yaml_info.md) - Detailed configuration file description
- [Final Model Training Architecture](docs/final_model_training_architecture.md) - Stage 2 model training
- [Threshold Setting Guide](docs/threshold_setting_guide.md) - Decision threshold selection

## Key Features Explained

### Nested Cross-Validation

metaClassifier uses strict nested cross-validation design:

1. **Outer CV**: Evaluates model generalization performance
2. **Inner CV**: Performs feature selection and hyperparameter tuning within each outer fold
3. **Data Separation**: Ensures strict separation of training, validation, and test data

### Consensus Feature Selection

Selects stable features through statistical consensus mechanism of inner CV:

- Each inner fold independently selects features
- Counts feature occurrence frequency across all inner folds
- Selects features with frequency above threshold as consensus feature set

### AUC Calculation Optimization

**Important Update**: v1.0 fixes AUC calculation method:

- **Old Method**: Calculate mean of per-fold AUC (inaccurate)
- **New Method**: Aggregate all outer_fold OOF predictions by repeat, calculate overall AUC for each repeat, then take the mean

This ensures accuracy and statistical significance of AUC calculation.

### Adaptive Variance Filtering

Dynamically adjusts variance filtering intensity based on data p/n ratio (features/samples):

- High-dimensional data (p>>n): Stricter filtering
- Low-dimensional data (p<n): More lenient filtering
- Configurable filtering parameters

## Output Results

### Build Command Output Structure

```
output/
├── 1_performance_metrics/        # Performance metrics
│   ├── nested_cv_pred_proba.csv # OOF prediction probabilities
│   ├── nested_cv_summary.csv     # Performance summary
│   └── ...
├── 2_final_model/                # Final model
│   ├── consensus_features.json   # Consensus feature set
│   ├── final_training_results.json
│   └── ...
├── 3_hyperparameter_analysis/    # Hyperparameter analysis
└── 4_reproducibility/            # Reproducibility information
    ├── run.log                   # Run log
    └── final_run.yaml            # Complete configuration
```

### Report Command Output Structure

```
output/reports/
├── within_disease/               # Within-disease analysis
├── between_project/              # Inter-project analysis
├── between_disease/               # Inter-disease analysis
├── overall/                      # Overall analysis
├── models/                       # Model comparison
└── predict_external_*/          # External prediction
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Environment Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/

# Type checking
mypy src/
```

## FAQ

### Q: How to choose the appropriate CV strategy?

**A:** 
- **K-Fold**: Suitable for large sample sizes with uniform cohort distribution
- **LOCO (Leave-One-Cohort-Out)**: Suitable when evaluating cross-cohort generalization ability
- **Repeated K-Fold**: Use when more stable performance estimation is needed

### Q: How to interpret consensus features?

**A:** Consensus features are features that stably appear across multiple inner CV folds, indicating these features have stable contributions to model performance.

### Q: Why is AUC calculation important?

**A:** The correct AUC calculation method (aggregated by repeat) can more accurately reflect the true performance of the model, avoiding bias caused by different sample sizes per fold.

## Changelog

### v1.0.0 (Current Version)

**Major Updates:**
- ✅ Fixed AUC calculation: Changed from per-fold AUC mean to overall AUC calculated by repeat
- ✅ Enhanced reporting system: Supports multiple analysis scenarios
- ✅ Optimized visualization: Supports repeat-mean ROC curves
- ✅ Enhanced reproducibility: Detailed recording of all experimental parameters

**New Features:**
- Support for Repeated K-Fold cross-validation
- Extended model support (CatBoost, XGBoost, etc.)
- Complete reporting pipeline
- Adaptive variance filtering optimization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

Thanks to all developers and researchers who contributed to this project.

## Contact

- **Issues**: [GitHub Issues](https://github.com/juyanmei/MetaClassifier/issues)
- **Documentation**: [Online Documentation](https://github.com/juyanmei/MetaClassifier)

---

**metaClassifier v1.0** - Making microbiome classification simpler, more reliable, and more reproducible 🧬🔬
