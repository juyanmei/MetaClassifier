import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

class FeatureImportanceCalculator:
    def __init__(self, models, random_seed=42):
        self.models = models
        self.random_seed = random_seed

    def get_feature_importance(self, X, y, feature_selected_dict=None, cv_folds=5):
        feature_importance_dict = {}
        print("Calculating Feature Importance...")

        for model_name, model in self.models.items():
            print(f"    For model {model_name} ")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
            print(f"    Using StratifiedKFold with {cv_folds} folds for cross-validation")
            importances = np.zeros(X.shape[1])

            # If feature_selected_dict exists, filter features
            if feature_selected_dict and model_name in feature_selected_dict:
                selected_features = feature_selected_dict[model_name]
                X_selected = X.iloc[:, selected_features]
            else:
                X_selected = X

            for train_idx, test_idx in cv.split(X_selected, y):
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)
                if feature_selected_dict is not None:
                    if model_name in {'RandomForest', 'XGBoost'}:
                        importances[selected_features] += model.feature_importances_
                    elif model_name == 'CatBoost':
                        importances[selected_features] += model.get_feature_importance()
                    elif model_name in {'LogisticRegression', 'lasso', 'ElasticNet', 'GMWI2'}:
                        importances[selected_features] += np.abs(model.coef_[0])
                    elif model_name in {'SVM', 'GaussianNB', 'MLP', 'NeuralNetwork'}:
                        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=self.random_seed)
                        importances[selected_features] += result.importances_mean
                else:
                    if model_name in {'RandomForest', 'XGBoost'}:
                        importances += model.feature_importances_
                    elif model_name == 'CatBoost':
                        importances += model.get_feature_importance()
                    elif model_name in {'LogisticRegression', 'lasso', 'ElasticNet'}:
                        importances += np.abs(model.coef_[0])
                    elif model_name in {'SVM', 'GaussianNB', 'MLP', 'NeuralNetwork'}:
                        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=self.random_seed)
                        importances += result.importances_mean

            importances /= cv.get_n_splits()
            feature_importance_dict[model_name] = importances
            #print(f"        {model_name} feature importances: {importances}")

        # Convert results to dataframe
        feature_importance_df = pd.DataFrame(feature_importance_dict, index=X.columns)
        return feature_importance_df
