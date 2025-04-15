from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from utils.PyTorch import SklearnPyTorchWrapper


class BaseModel:
    def __init__(self, models=None, random_seed=42):
        self.random_seed = random_seed
        if models is None:
            models = ['RandomForest', 'CatBoost', 'LogisticRegression', 'MLP', 'XGBoost', 'lasso', 'SVM', 'ElasticNet', 'NeuralNetwork', 'GaussianNB', 'KNN', 'GMWI2']
        self.models = self.initialize_methods(models)
        print("Model Initializing...")
        print(f"The model : {models}")
        print("-----------------------------------")
        print()
        
    def initialize_methods(self, models):
        method_dict = {
            'RandomForest': RandomForestClassifier(random_state=self.random_seed),
            'CatBoost': CatBoostClassifier(random_seed=self.random_seed, silent=True),
            'LogisticRegression': LogisticRegression(),
            'MLP': MLPClassifier(random_state=self.random_seed),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=self.random_seed),
            'lasso': LogisticRegression(C=10, penalty='l1', solver='liblinear', class_weight="balanced", random_state=self.random_seed),  
            'SVM': SVC(kernel='linear', probability=True),
            'NeuralNetwork': SklearnPyTorchWrapper(epochs=10, batch_size=32, learning_rate=0.01, random_seed=self.random_seed), 
            'ElasticNet': LogisticRegression(penalty='elasticnet', l1_ratio=0.01, solver='saga', random_state=self.random_seed),  # Logistic regression with ElasticNet regularization
            'GaussianNB': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'GMWI2': LogisticRegression(C=0.03, penalty='l1', solver='liblinear', class_weight="balanced", random_state=self.random_seed)
        }
        return {model: method_dict[model] for model in models if model in method_dict}

