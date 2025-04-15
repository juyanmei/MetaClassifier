from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

class ModelEvaluator:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def precision_with_zero_division(self, y_true, y_pred):
        try:
            return precision_score(y_true, y_pred, zero_division=1)
        except Exception as e:
            # You can choose to print the error, log it, or handle it in other ways
            print(f"Error calculating precision: {e}")
            return 0  # Or return np.nan, depending on how you want to handle this situation
            
    def specificity_score(self, y_true, y_pred):
        """Calculate specificity (true negative rate)"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            if (tn + fp) == 0:
                print("000000")
                return 0  # Or return np.nan, depending on how you want to handle this situation
            return tn / (tn + fp)
        except Exception as e:
            print(f"Error in specificity_score: {e}")
            return np.nan


    def evaluate_models(self, models, X, y, y_cohort=None, feature_dict=None, n_repeats=1, cpu=8, cv_folds=5):
        print("Model Evaluating...")
        print(f"    For {models}...")
        
        if y_cohort is not None:
            # Using LeaveOneGroupOut for cross-validation
            o = OrdinalEncoder()
            groups = o.fit_transform(y_cohort.reshape((len(y_cohort), 1))).flatten()
            cv = LeaveOneGroupOut()
            print("    Using LeaveOneGroupOut for cross-validation")
            if n_repeats > 1:
                print("    Warning: n_repeats parameter is ignored when using LeaveOneGroupOut as it's deterministic")
        else:
            # Using StratifiedKFold for cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, random_state=self.random_seed, shuffle=True)
            print(f"    Using StratifiedKFold with {cv_folds} folds for cross-validation")
        
        scoring = {
            'test_accuracy': 'accuracy',
            'test_precision': make_scorer(self.precision_with_zero_division),
            'test_recall': 'recall',
            'test_f1': 'f1',
            'test_roc_auc': 'roc_auc',
            'test_mcc': make_scorer(matthews_corrcoef),
            'test_specificity': make_scorer(self.specificity_score)
        }
        
        model_scores = {}
        model_predictions = pd.DataFrame(index=X.index)
        model_predictions['true_values'] = y
        # Add whole_predictions to store raw prediction results for each repeat, using DataFrame format consistent with model_predictions
        whole_predictions = pd.DataFrame(index=X.index)
        whole_predictions['true_values'] = y
        # internal_val information will be directly stored in scores
        
        for name, model in models.items():
            if feature_dict is not None:
                X_new =  X.iloc[:, feature_dict[name]]
            else:
                X_new = X
            
            all_predictions = np.zeros(len(y))
            all_scores = {metric: [] for metric in scoring.keys()}
            all_scores['fit_time'] = []
            all_scores['score_time'] = []
            all_scores['Repeat'] = []
            
            try:
                if y_cohort is not None:
                    if n_repeats > 1:
                        print("    Warning: n_repeats parameter is ignored when using LeaveOneGroupOut as it's deterministic")
                    # Only execute once when using LeaveOneGroupOut
                    scores = cross_validate(model, X_new, y, cv=cv, groups=groups, scoring=scoring, return_train_score=False, n_jobs=cpu)
                    y_pred_proba = cross_val_predict(model, X_new, y, cv=cv, groups=groups, method='predict_proba', n_jobs=cpu)
                    # Record the currently excluded group name
                    unique_groups = np.unique(groups)
                    
                    # Accumulate scoring results
                    for metric in scoring.keys():
                        all_scores[metric].extend(scores[f'test_{metric}'])
                    all_scores['fit_time'].extend(scores['fit_time'])
                    all_scores['score_time'].extend(scores['score_time'])
                    # Add group information to scoring results
                    all_scores['CV'] = [str(y_cohort[groups == group][0]) for group in unique_groups]
                    all_scores['Repeat'] = [1] * len(unique_groups)
                    all_scores['internal_val'] = [str(group) for group in unique_groups]
                    all_predictions = y_pred_proba[:, 1]
                    
                    # For LeaveOneGroupOut, add prediction results to whole_predictions DataFrame
                    whole_predictions[f'{name}_prediction'] = y_pred_proba[:, 1]
                    whole_predictions[f'{name}_repeat'] = 1  # Only one repeat
                else:
                    # Execute n_repeats times when using StratifiedKFold
                    
                    for i in range(n_repeats):
                        print(f"Repeat {i+1}/{n_repeats}")
                        # Use different random seeds for each repeat to ensure true multiple evaluations
                        current_seed = self.random_seed + i
                        current_cv = StratifiedKFold(n_splits=cv_folds, random_state=current_seed, shuffle=True)
                        scores = cross_validate(model, X_new, y, cv=current_cv, scoring=scoring, return_train_score=False, n_jobs=cpu)
                        y_pred_proba = cross_val_predict(model, X_new, y, cv=current_cv, method='predict_proba', n_jobs=cpu)
                        
                        # Accumulate scoring results for each repeat
                        for metric in scoring.keys():
                            all_scores[metric].extend(scores[f'test_{metric}'])
                        all_scores['fit_time'].extend(scores['fit_time'])
                        all_scores['score_time'].extend(scores['score_time'])
                        all_scores['Repeat'].extend([i+1] * cv_folds)
                        all_predictions += y_pred_proba[:, 1]
                        
                        # Save raw prediction results for each repeat to whole_predictions DataFrame
                        whole_predictions[f'{name}_prediction_repeat{i+1}'] = y_pred_proba[:, 1]
                    
                    # Take the average
                    all_predictions /= n_repeats
                    
                    # Add group information to scoring results
                    all_scores['CV'] = [f'Fold{(i%cv_folds)+1}' for i in range(cv_folds * n_repeats)]
                    
                    # Calculate average prediction results across all repeats
                    whole_predictions[f'{name}_prediction_avg'] = all_predictions
                
                # Set prediction values
                model_predictions[name] = all_predictions
                
                model_scores[name] = all_scores
            
            except Exception as e:
                print(f"Error in evaluate_models for model {name}: {e}")
                model_scores[name] = {metric: np.nan for metric in scoring.keys()}
                model_scores[name]['fit_time'] = np.nan
                model_scores[name]['score_time'] = np.nan
        
        print("Model Evaluation Finished!!!")
        print("----------------------------------")
        print()
        return model_scores, model_predictions, whole_predictions


    def evaluate_models_stacking(self, stacking_clf, X, y, y_cohort=None, feature_dict=None, n_repeats=1, cpu=8, cv_folds=5):
        """评估多个模型的效果 （还需要修改）"""
        print("Model Evaluating...")
        print(f"    For StackingClassifier...")
        
        if y_cohort is not None:
            # Using LeaveOneGroupOut for cross-validation
            o = OrdinalEncoder()
            groups = o.fit_transform(y_cohort.reshape((len(y_cohort), 1))).flatten()
            cv = LeaveOneGroupOut()
            print("    Using LeaveOneGroupOut for cross-validation")
            if n_repeats > 1:
                print("    Warning: n_repeats parameter is ignored when using LeaveOneGroupOut as it's deterministic")
        else:
            # Using StratifiedKFold for cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, random_state=self.random_seed, shuffle=True)
            print(f"    Using StratifiedKFold with {cv_folds} folds for cross-validation")
        
        scoring = {
            'test_accuracy': 'accuracy',
            'test_precision': make_scorer(self.precision_with_zero_division),
            'test_recall': 'recall',
            'test_f1': 'f1',
            'test_roc_auc': 'roc_auc',
            'test_mcc': make_scorer(matthews_corrcoef),
            'test_specificity': make_scorer(self.specificity_score)
        }
        model_scores = {}
        model_predictions = pd.DataFrame(index=X.index)
        model_predictions['true_values'] = y
        # Add whole_predictions to store raw prediction results for each repeat, using DataFrame format consistent with model_predictions
        whole_predictions = pd.DataFrame(index=X.index)
        whole_predictions['true_values'] = y
        # internal_val information will be directly stored in scores
        
        try:
            if y_cohort is not None:
                if n_repeats > 1:
                    print("    Warning: n_repeats parameter is ignored when using LeaveOneGroupOut as it's deterministic")
                
                # Only execute once when using LeaveOneGroupOut
                scores = cross_validate(stacking_clf, X, y, cv=cv, groups=groups, scoring=scoring, return_train_score=False, n_jobs=cpu)
                
                # Perform cross-validation to get predictions
                meta_features = np.zeros((X.shape[0], len(feature_dict)))
                final_predictions = np.zeros(len(y))
                
                splits = cv.split(X, y, groups=groups)
                for train_index, test_index in splits:
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    # 训练基学习器并获取预测
                    for (name, model) in stacking_clf.estimators:
                        X_train_selected = X_train.iloc[:, feature_dict[name]]
                        X_test_selected = X_test.iloc[:, feature_dict[name]]
                        model.fit(X_train_selected, y_train)
                        y_pred = model.predict_proba(X_test_selected)[:, 1]
                        meta_features[test_index, list(feature_dict.keys()).index(name)] = y_pred
                    
                    # 训练元学习器并获取预测
                    stacking_clf.final_estimator.fit(meta_features[train_index], y_train)
                    final_predictions[test_index] = stacking_clf.final_estimator.predict_proba(meta_features[test_index])[:, 1]
                
                model_predictions['StackingClassifier'] = final_predictions
                
                # 将预测结果添加到whole_predictions DataFrame中
                whole_predictions['StackingClassifier_prediction'] = final_predictions
                whole_predictions['StackingClassifier_repeat'] = 1  # 只有一次repeat
                
                # Record scoring results
                model_scores['StackingClassifier'] = {}
                for metric in scoring.keys():
                    model_scores['StackingClassifier'][metric] = scores[f'test_{metric}']
                model_scores['StackingClassifier']['fit_time'] = scores['fit_time']
                model_scores['StackingClassifier']['score_time'] = scores['score_time']
                model_scores['StackingClassifier']['internal_val'] = [str(group) for group in groups]
                
            else:
                # Execute n_repeats times when using StratifiedKFold
                all_final_predictions = np.zeros(len(y))
                all_scores = {metric: [] for metric in scoring.keys()}
                all_scores['fit_time'] = []
                all_scores['score_time'] = []
                
                for i in range(n_repeats):
                    print(f"Repeat {i+1}/{n_repeats}")
                    # 为每次重复使用不同的随机种子
                    current_seed = self.random_seed + i
                    current_cv = StratifiedKFold(n_splits=cv_folds, random_state=current_seed, shuffle=True)
                    
                    # Perform cross-validation to get predictions
                    meta_features = np.zeros((X.shape[0], len(feature_dict)))
                    final_predictions = np.zeros(len(y))
                    
                    splits = current_cv.split(X, y)
                    for train_index, test_index in splits:
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        # Train base learners and get predictions
                        for (name, model) in stacking_clf.estimators:
                            X_train_selected = X_train.iloc[:, feature_dict[name]]
                            X_test_selected = X_test.iloc[:, feature_dict[name]]
                            model.fit(X_train_selected, y_train)
                            y_pred = model.predict_proba(X_test_selected)[:, 1]
                            meta_features[test_index, list(feature_dict.keys()).index(name)] = y_pred
                        
                        # Train meta-learner and get predictions
                        stacking_clf.final_estimator.fit(meta_features[train_index], y_train)
                        final_predictions[test_index] = stacking_clf.final_estimator.predict_proba(meta_features[test_index])[:, 1]
                    
                    # 保存每次repeat的预测结果到whole_predictions DataFrame中
                    whole_predictions[f'StackingClassifier_prediction_repeat{i+1}'] = final_predictions
                    
                    all_final_predictions += final_predictions
                    
                    # 获取评分结果
                    scores = cross_validate(stacking_clf, X, y, cv=current_cv, scoring=scoring, return_train_score=False, n_jobs=cpu)
                    
                    # Accumulate scoring results for each repeat
                    for metric in scoring.keys():
                        all_scores[metric].extend(scores[f'test_{metric}'])
                    all_scores['fit_time'].extend(scores['fit_time'])
                    all_scores['score_time'].extend(scores['score_time'])
                
                # Take the average
                all_final_predictions /= n_repeats
                model_predictions['StackingClassifier'] = all_final_predictions
                
                # 添加平均预测结果
                whole_predictions['StackingClassifier_prediction_avg'] = all_final_predictions
                
                # Record scoring results
                model_scores['StackingClassifier'] = all_scores
                model_scores['StackingClassifier']['LOCO'] = ['NA'] * (cv_folds * n_repeats)
            
        except Exception as e:
            print(f"Error in evaluate_models_stacking: {e}")
            model_scores['StackingClassifier'] = {metric: np.nan for metric in scoring.keys()}
            model_scores['StackingClassifier']['fit_time'] = np.nan
            model_scores['StackingClassifier']['score_time'] = np.nan
        
        print("Model Evaluation Finished!!!")
        print("----------------------------------")
        print()
        return model_scores, model_predictions, whole_predictions

    def evaluate_model_nocv(self, model, model_name, X, y):
        # Predict test set results
        y_pred = model.predict(X)
        # Calculate various scoring metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)  # zero_division=0 handles division by zero cases
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
        specificity = self.specificity_score(y, y_pred)
        
        # Store scoring metrics in a dictionary
        scores = {
            'Model' : model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC_AUC': roc_auc,
            'MCC': mcc,
            'Specificity': specificity
        }
        df_scores = pd.DataFrame(scores, index=[0])
        # Return scoring metrics
        return df_scores



