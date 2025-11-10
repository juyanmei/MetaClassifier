"""
二分类任务专用配置 for metaClassifier v1.0.

针对二分类任务优化的配置参数。
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class BinaryClassificationConfig:
    """二分类任务配置类。"""
    
    # 评估指标配置
    primary_metric: str = "roc_auc"  # 主要评估指标
    secondary_metrics: List[str] = None  # 次要评估指标
    
    # 类别不平衡处理
    handle_imbalance: bool = True  # 是否处理类别不平衡
    imbalance_threshold: float = 0.3  # 不平衡阈值
    imbalance_strategies: List[str] = None  # 不平衡处理策略
    
    # 超参数搜索配置
    hyperparameter_search: Dict[str, Any] = None
    
    # 交叉验证配置
    cv_strategy: str = "stratified"  # CV策略
    cv_folds: int = 5  # CV折数
    cv_repeats: int = 1  # CV重复次数
    
    def __post_init__(self):
        """初始化后处理。"""
        if self.secondary_metrics is None:
            self.secondary_metrics = [
                "precision", "recall", "f1", "specificity", 
                "balanced_accuracy", "matthews_corrcoef", "average_precision"
            ]
        
        if self.imbalance_strategies is None:
            self.imbalance_strategies = ["class_weight", "sampling", "threshold_adjustment"]
        
        if self.hyperparameter_search is None:
            self.hyperparameter_search = {
                "method": "grid",  # 搜索方法
                "n_iter": 50,  # 随机搜索迭代次数
                "cv": 3,  # 内层CV折数
                "scoring": "combined_binary",  # 评分策略
                "n_jobs": -1  # 并行作业数
            }


# 二分类任务专用模型配置
BINARY_CLASSIFICATION_MODEL_CONFIGS = {
    "Lasso": {
        "hyperparameters": {
            "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "max_iter": [2000, 5000, 10000],
            "tol": [1e-5, 1e-4, 1e-3],
            "solver": ["liblinear", "saga"],
            "class_weight": ["balanced", None],
            "random_state": [42]
        },
        "feature_selection": {
            "method": "selectfrommodel",
            "max_features": 100,
            "threshold": "median"
        }
    },
    
    "RandomForest": {
        "hyperparameters": {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.1, 0.2],
            "bootstrap": [True, False],
            "class_weight": ["balanced", "balanced_subsample", None],
            "max_samples": [0.8, 0.9, 1.0],
            "random_state": [42]
        },
        "feature_selection": {
            "method": "selectfrommodel",
            "max_features": 200,
            "threshold": "1.25*median"
        }
    },
    
    "XGBoost": {
        "hyperparameters": {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [3, 6, 9, 12],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "scale_pos_weight": [1, 2, 3, 5],  # 处理类别不平衡
            "random_state": [42]
        },
        "feature_selection": {
            "method": "selectfrommodel",
            "max_features": 300,
            "threshold": "1.5*median"
        }
    },
    
    "SVM": {
        "hyperparameters": {
            "kernel": ["linear", "rbf", "poly"],
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "class_weight": ["balanced", None],
            "probability": [True],  # 需要概率预测
            "random_state": [42]
        },
        "feature_selection": {
            "method": "selectkbest",
            "max_features": 50,
            "k": 50
        }
    },
    
    "NeuralNetwork": {
        "hyperparameters": {
            "hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (200, 100)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "lbfgs"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "max_iter": [500, 1000, 2000],
            "early_stopping": [True, False],
            "random_state": [42]
        },
        "feature_selection": {
            "method": "selectkbest",
            "max_features": 100,
            "k": 100
        }
    }
}


# 二分类任务评估指标配置
BINARY_CLASSIFICATION_METRICS = {
    "primary": "roc_auc",
    "secondary": [
        "accuracy", "precision", "recall", "f1", 
        "specificity", "balanced_accuracy", "matthews_corrcoef",
        "average_precision", "log_loss"
    ],
    "threshold_metrics": [
        "precision", "recall", "f1", "specificity", "balanced_accuracy"
    ],
    "probability_metrics": [
        "roc_auc", "average_precision", "log_loss"
    ]
}


# 类别不平衡处理策略
IMBALANCE_HANDLING_STRATEGIES = {
    "class_weight": {
        "lasso": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}],
        "randomforest": ["balanced", "balanced_subsample", {0: 1, 1: 2}, {0: 1, 1: 3}],
        "svm": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        "neuralnetwork": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}]
    },
    "sampling": {
        "oversampling": ["SMOTE", "ADASYN", "BorderlineSMOTE"],
        "undersampling": ["RandomUnderSampler", "EditedNearestNeighbours"],
        "combined": ["SMOTEENN", "SMOTETomek"]
    },
    "threshold_adjustment": {
        "methods": ["optimal", "youden", "f1_optimal"],
        "threshold_range": [0.1, 0.9]
    }
}


def get_binary_classification_config(
    model_name: str, 
    data_shape: tuple = None,
    class_balance: float = None
) -> Dict[str, Any]:
    """
    获取针对二分类任务优化的模型配置。
    
    Args:
        model_name: 模型名称
        data_shape: 数据形状 (n_samples, n_features)
        class_balance: 类别平衡比例
        
    Returns:
        优化的模型配置
    """
    config = BINARY_CLASSIFICATION_MODEL_CONFIGS.get(model_name, {})
    
    # 根据数据特征调整配置
    if data_shape and len(data_shape) == 2:
        n_samples, n_features = data_shape
        
        # 高维数据调整
        if n_features > 1000:
            if "max_features" in config.get("hyperparameters", {}):
                config["hyperparameters"]["max_features"] = ["sqrt", "log2", 0.05, 0.1]
            if "max_features" in config.get("feature_selection", {}):
                config["feature_selection"]["max_features"] = min(100, n_features // 10)
        
        # 小样本数据调整
        if n_samples < 100:
            if "n_estimators" in config.get("hyperparameters", {}):
                config["hyperparameters"]["n_estimators"] = [50, 100, 200]
            if "max_depth" in config.get("hyperparameters", {}):
                config["hyperparameters"]["max_depth"] = [5, 10, 15]
    
    # 根据类别不平衡调整配置
    if class_balance is not None and class_balance < 0.3:
        # 严重不平衡，添加更多的不平衡处理参数
        if "class_weight" in config.get("hyperparameters", {}):
            # 添加更激进的不平衡处理
            config["hyperparameters"]["class_weight"].extend([
                {0: 1, 1: 5}, {0: 1, 1: 10}
            ])
    
    return config
