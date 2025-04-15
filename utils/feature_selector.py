from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

class FeatureSelector:
    def __init__(self, models, random_seed=42):
        self.models = models
        self.random_seed = random_seed

    def pre_feature_selection(self, X):
        # Only for testing
        # Initialize variance threshold selector with threshold 0.0005
        selector = VarianceThreshold(threshold=0.0005)
        # Select features
        X_selected = selector.fit_transform(X)
        # Record original feature indices
        original_indices = np.arange(X.shape[1])
        # Get indices of retained features
        retained_indices = original_indices[selector.get_support()]
        print("保留的特征原始列索引:", retained_indices)
        return retained_indices
    
    def calculate_feature_correlation(self, data, threshold=0.7, p_value_threshold=0.05):
        """
        计算特征之间的Spearman相关性，并根据阈值删除高度相关的特征。

        参数:
        - data (pd.DataFrame): 包含特征的数据框。
        - threshold (float): 相关性阈值，默认值为0.7。
        - p_value_threshold (float): p值阈值，默认值为0.05。

        返回:
        - original_corr_matrix (pd.DataFrame): 删除特征前的自相关矩阵。
        - reduced_corr_matrix (pd.DataFrame): 删除特征后的自相关矩阵。
        - removed_features (list): 被删除特征的索引列表。
        """
        # Calculate autocorrelation matrix before feature removal
        original_corr_matrix = data.corr(method='spearman')
        np.random.seed(42)
        # Initialize list of features to remove
        removed_features = []
        removed_features_indices = []
        # Create a copy to avoid modifying original data
        data_copy = data.copy()

        # Calculate Spearman correlation and p-values between features
        for i in range(len(data.columns)):
            for j in range(i + 1, len(data.columns)):
                feature_1 = data.columns[i]
                feature_2 = data.columns[j]
                corr, p_value = spearmanr(data[feature_1], data[feature_2])
                if abs(corr) > threshold and p_value < p_value_threshold:
                    # Randomly remove one of the features
                    feature_to_remove = np.random.choice([feature_1, feature_2])
                    if feature_to_remove not in removed_features:
                        removed_features.append(feature_to_remove)
                        removed_features_indices.append(data.columns.get_loc(feature_to_remove))
                        data_copy = data_copy.drop(columns=[feature_to_remove])

        # Calculate autocorrelation matrix after feature removal
        reduced_corr_matrix = data_copy.corr(method='spearman')

        return data_copy, removed_features_indices, removed_features

    def select_feature(self, X, y, calculate_correlation=True, cv_folds=5):
        selected_features_dict = {}
        print("Feature Selecting...")
        for model_name, model in self.models.items():
            print(f"    For model {model_name} : ")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle = True, random_state=self.random_seed)
            print(f"    Using StratifiedKFold with {cv_folds} folds for cross-validation")
            selected_indices_list = []

            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                if model_name in {'SVM', 'GaussianNB', 'MLP', 'NeuralNetwork', 'KNN'}:
                    selected_indices = self.pre_feature_selection(X_train)
                    #selected_indices = np.arange(0, X.shape[1])
                    print(selected_indices)
                    selected_indices_list.append(set(selected_indices))
                    print(f'    Model {model_name} without feature selecting')
                else:
                    selector = SelectFromModel(model, max_features=20)
                    selector.fit(X_train, y_train)
                    selected_indices = selector.get_support(indices=True)
                    selected_indices_list.append(set(selected_indices))
                    print(f"        {model_name} selected_indices : {selected_indices}")        
            # Calculate intersection of all folds
            model_selected_indices = sorted(list(set.union(*selected_indices_list)))
            if calculate_correlation:
                # Convert selected features to DataFrame
                print("        Correlation Calculating...")
                #selected_features_df = pd.DataFrame(X_new, columns=[f'feature_{i}' for i in final_selected_indices])
                selected_features_df = X.iloc[:, model_selected_indices]
                reduced_X, removed_features_indices, removed_features = self.calculate_feature_correlation(selected_features_df)
                print(f"            Removed Features for {model_name}: {removed_features}")
                print(f"                after {reduced_X.shape}")
                for pos in sorted(removed_features_indices, reverse=True):
                    del model_selected_indices[pos]
                print("        Correlation done...")
            selected_features_dict[model_name] = model_selected_indices
            print(f"    union of Selected Feature indices : {model_name} : {len(model_selected_indices)} : ")
            print(f"        {model_selected_indices}") 

        print(f"selected_feature : {selected_features_dict}")
        print("Feature Selection Finished!!!")
        print()
        print("-----------------------------------")
        return selected_features_dict
