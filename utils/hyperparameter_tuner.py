from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
import os
import warnings
import logging

# Configure warning handling
def log_warnings_to_file(message, category, filename, lineno, file=None, line=None):
    # Configure warning logger handler
    warning_logger = logging.getLogger('warning_logger')
    warning_logger.setLevel(logging.WARNING)  # Only handle WARNING level and above
    warning_logger.propagate = False  # Prevent propagation to other handlers
    
    # Ensure handlers exist
    if not warning_logger.handlers:
        # Create file handler
        file_handler = logging.FileHandler('warning.log', mode='a', encoding='utf-8')
        file_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        warning_logger.addHandler(file_handler)
    
    warning_logger.warning(f"{filename}:{lineno} - {category.__name__}: {message}")
    for handler in warning_logger.handlers:
        handler.flush()

warnings.showwarning = log_warnings_to_file

class HyperparameterTuner:
    def __init__(self, param_grids=None, random_seed=42):
        """
        Initialize hyperparameter tuner.

        Parameters:
        param_grids (dict): Mapping from model names to parameter grids.
        random_seed (int): Random seed.
        """
        if param_grids is None:
            param_grids = {
                'RandomForest': {
                    'n_estimators': [100, 200, 300, 500, 1000, 1500],
                    'max_depth': [None, 10, 20, 30, 50, 100],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                },
                'CatBoost': {
                    'iterations': [500, 1000, 1500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'depth': [3, 4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9]
                },
                'LogisticRegression': {
                    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
                    'penalty': [None],
                    'max_iter': [500, 1000, 5000],
                },
                'MLP': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (50, 100, 50)],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'max_iter': [500, 1000, 1500],
                    'solver': ['adam', 'lbfgs', 'sgd'],
                    'activation': ['identity', 'logistic', 'tanh', 'relu']
                },
                'lasso': {
                    'max_iter': [500, 1000, 5000],
                    'C': [0.001, 0.005, 0.01, 0.1, 1, 10, 100],
                    'solver': ['saga', 'liblinear'],
                    'tol': [0.0001, 0.001, 0.01]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 7, 8, 10],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                'SVM': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3, 4]
                },
                'NeuralNetwork': {
                    'epochs': [10, 20, 30, 50],
                    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                    'batch_size': [4, 8, 16, 32, 64],
                    'optimizer': ['adam', 'sgd', 'rmsprop']
                },
                'ElasticNet': {
                    'l1_ratio': [0.01, 0.05, 0.1, 0.5, 0.9],
                    'C': [0.1, 1.0, 5.0, 10.0, 50.0],
                    'max_iter': [500, 1000, 5000],
                    'solver': ['saga'],
                    'tol': [0.0001, 0.001, 0.01]
                },
                'GaussianNB': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
                'KNN': {
                    'n_neighbors': range(3, 21),
                    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }
        self.param_grids = param_grids
        self.random_seed = random_seed
        
        # Capture all warnings
        warnings.filterwarnings("always")
        warnings.simplefilter("always", ConvergenceWarning)

    def tune_model(self, model_name, model, X, y, cpu=8, y_cohort=None, feature=None):
        """
        Tune hyperparameters for the specified model.

        Parameters:
        model_name (str): Model name.
        model (sklearn.base.BaseEstimator): Model instance to tune.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Label vector.
        y_cohort (numpy.ndarray): Cohort label vector.

        Returns:
        best_estimator (sklearn.base.BaseEstimator): Model instance with optimal hyperparameters.
        best_params (dict): Optimal hyperparameters.
        """
        param_grid = self.param_grids.get(model_name, {})
        if param_grid:
            print("HyperparameterTuner Start!!!")
            print(f"    The model is: {model}")
            if feature is not None:
                X = X.iloc[:, feature]
                print("    For selected features")
            try:
                if y_cohort is not None:
                    # Define LeaveOneGroupOut GridSearchCV for hyperparameter search
                    o = OrdinalEncoder()
                    groups = o.fit_transform(y_cohort.reshape((len(y_cohort), 1))).flatten()
                    logo = LeaveOneGroupOut()
                    scorer = make_scorer(balanced_accuracy_score)
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=logo, n_jobs=cpu, verbose=1)
                    grid_search.fit(X, y, groups=groups)
                    print("    LeaveOneGroupOut GridSearchCV")

                else:
                    scorer = make_scorer(balanced_accuracy_score)
                    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, n_jobs=cpu)
                    grid_search.fit(X, y)
                    print("    Basic GridSearchCV")                        
                                
                print("    The best params : ")
                print(f"        {grid_search.best_estimator_}")
                print("HyperparameterTuner Finished!!!")
                print()
                print("-----------------------------------")
                return grid_search.best_estimator_, grid_search.best_params_
            except Exception as e:
                print(f"Error tuning {model_name}: {e}")
                return model, {}
        
