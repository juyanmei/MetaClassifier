from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.svm import SVC

class StackingModel:
    def __init__(self, base_models, random_seed=42):
        self.base_models = base_models
        self.random_seed = random_seed
        
    def get_stacking_simple(self):
        estimators = [(name, model) for name, model in self.base_models.items()]
        stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
        return stacking_clf

    def get_stacking_auto(self, models, model_scores, model_predictions, auc_threshold=0.7, cpu=8):
        if len(models) > 1:
            estimators = self.select_stacking_base_models(models, model_scores, model_predictions, auc_threshold)
            if len(estimators) > 1:
                final_estimator = RandomForestClassifier(n_estimators=1000, random_state=self.random_seed)
                stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=cpu)
                return stacking_clf
            else:
                return estimators
        else:
            return models

    
    def select_stacking_base_models(self, models, model_scores, model_predictions, auc_threshold=0.7):
        """
        选择最适合的基模型用于stacking。
        
        参数:
        model_score (dict): 包含模型名称和对应AUC分数的字典。
        model_predictions (pd.DataFrame): 包含模型名称和对应预测值的数据框。
        auc_threshold (float): 初步筛选的AUC阈值。
        
        返回:
        list: 最终选择的模型名称和实例的列表。
        """
        
        # 第一步：筛选出平均AUC > auc_threshold的模型
        model_auc = {model: scores['test_roc_auc'] for model, scores in model_scores.items() if 'test_roc_auc' in scores}
        model_auc_mean = {model: sum(aucs) / len(aucs) for model, aucs in model_auc.items()}
        selected_models = [name for name, score in model_auc_mean.items() if score > auc_threshold]
        print(f"Models with AUC :  {model_auc_mean}")
        print("Models with AUC >", auc_threshold, ":", selected_models)

        if len(selected_models) > 1:
            # 第二步：计算筛选后模型的相关性并选择相关性较低的模型
            predictions = model_predictions[selected_models]
            correlation_matrix = predictions.corr()
            print("Correlation matrix of selected models:")
            print(correlation_matrix)
            
            # 保留单个模型得分最高的模型A
            sorted_models = sorted(model_auc_mean, key=model_auc_mean.get, reverse=True)
            model_A = sorted_models[0]
            
            # 选择与模型A相关性最低的模型B
            correlations_A = correlation_matrix[model_A].drop(model_A)
            model_B = correlations_A.idxmin()
            
            # 保留单个模型得分排名第二高的模型C
            model_C = sorted_models[1]
            
            # 选择与模型C相关性最低的模型D
            correlations_C = correlation_matrix[model_C].drop(model_C)
            model_D = correlations_C.idxmin()
            
            # 最终模型A、B、C、D取并集
            final_model_names = set([model_A, model_B, model_C, model_D])
            final_base_models = [(name, models[name]) for name in final_model_names]
        else:
            # 保留单个模型得分最高的模型A
            sorted_models = sorted(model_auc_mean, key=model_auc_mean.get, reverse=True)
            model_A = sorted_models[0]
            final_model_names = set([model_A])
            final_base_models = [(name, models[name]) for name in final_model_names]
        print("Final models for stacking:", final_base_models)
        
        return final_base_models
