# Standard library imports
import argparse
import csv
import os
import sys

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                   cross_val_score)

# Local module imports
from utils.analyze_results import (analyze_predictions, find_optimal_cutoff,
                                  generate_boxplots)
from utils.base_model import BaseModel
from utils.data_utils import (convert_scores_to_dataframe, filter_data,
                             load_and_prepare_data, remove_duplicate_columns,
                             save_models, save_results, setup_logging)
from utils.featureImportance import FeatureImportanceCalculator
from utils.feature_selector import FeatureSelector
from utils.hyperparameter_tuner import HyperparameterTuner
from utils.model_evaluator import ModelEvaluator
from utils.stacking_model import StackingModel

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add utils folder path to sys.path
sys.path.append(os.path.join(current_dir, 'utils'))


def str2bool(value):
    if isinstance(value, str):
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')



__version__ = "0.1.0"

def main(args):
    if args.target_id:
        directory = f"{args.output}/{args.target}/{args.target_id}"
    else:
        directory = f"{args.output}/{args.target}"
    os.makedirs(directory, exist_ok=True)

    # Setup logging
    logger = setup_logging(directory)
    
    # Load and prepare data using functions
    prof, metadata, cohorts = load_and_prepare_data(args.prof_file, args.metadata_file, args.target)
    X, y, y_cohort = filter_data(prof, metadata, args.target, args.target_id, args.y_cohort_group, args.y_cohort_type, args.pres)
    
    # Print first 5 rows and 5 columns of data
    print(X.iloc[:5, :5])

    if y_cohort is not None:
        print(f"Unique y_cohort values: {np.unique(y_cohort)}")
    else:
        print("not apply LOCO, y_cohort is None")
        
    # Define models to use, whether to perform feature selection and parameter tuning
    models = args.models
    perform_feature_selection = args.feature
    perform_tuner_param = args.turner
    ensemble = args.ensemble
    cpu = args.cpu
    repeat = args.repeat
    cv_folds = args.cv

    # Create base model instances
    base_model = BaseModel(models=models)
    base_models = base_model.models

    # Whether to perform feature selection
    if perform_feature_selection:
        selector = FeatureSelector(base_models)
        selected_features = selector.select_feature(X, y, calculate_correlation=True, cv_folds=cv_folds)
    else:
        selected_features = None


    # Whether to tune parameters
    if perform_tuner_param:
        tuner = HyperparameterTuner()
        for model_name in base_models:
            if model_name in tuner.param_grids:
                if perform_feature_selection:
                    best_model, best_params = tuner.tune_model(model_name, base_models[model_name], X, y, cpu=cpu, y_cohort=y_cohort, feature=selected_features[model_name])
                else:
                    best_model, best_params = tuner.tune_model(model_name, base_models[model_name], X, y, cpu=cpu, y_cohort=y_cohort)
                base_models[model_name] = best_model

    # Create model evaluator instance
    evaluator = ModelEvaluator()
    base_model_scores, base_model_predictions, base_model_whole_predictions = evaluator.evaluate_models(base_models, X, y, y_cohort=y_cohort, feature_dict=selected_features, n_repeats=repeat, cv_folds=cv_folds)

    
    # Calculate feature importance
    calculator = FeatureImportanceCalculator(base_models)
    # Calculate feature importance
    feature_importance_df = calculator.get_feature_importance(X, y, feature_selected_dict=selected_features, cv_folds=cv_folds)

    # Whether to build ensemble model
    if ensemble:
        # Automatically select base_model and meta_model:
        stacking_model = StackingModel(base_models)
        stacking_clf = stacking_model.get_stacking_auto(base_models, base_model_scores, base_model_predictions, 
                                                        auc_threshold=0.6, cpu=cpu)
        if isinstance(stacking_clf, StackingClassifier):
            final_base_models = stacking_clf.get_params()['estimators']
            model_names = [name for name, _ in final_base_models]
            filtered_base_model_scores = {model: scores for model, scores in base_model_scores.items() if model in model_names}
            if selected_features is not None:
                stacking_model_scores, stacking_model_predictions, stacking_model_whole_predictions = evaluator.evaluate_models_stacking(stacking_clf, X, y, feature_dict=selected_features, y_cohort=y_cohort, n_repeats=repeat, cv_folds=cv_folds)
            else:
                stacking_model_scores, stacking_model_predictions, stacking_model_whole_predictions = evaluator.evaluate_models({'StackingClassifier': stacking_clf}, X, y, y_cohort=y_cohort, n_repeats=repeat, cv_folds=cv_folds)
            # Merge base_model and stacking_model
            all_scores = {**base_model_scores, **stacking_model_scores}
            all_predictions = remove_duplicate_columns(pd.concat([base_model_predictions, stacking_model_predictions], axis=1))
            all_whole_predictions = remove_duplicate_columns(pd.concat([base_model_whole_predictions, stacking_model_whole_predictions], axis=1))
        else:
            all_scores = base_model_scores
            all_predictions = base_model_predictions
            all_whole_predictions = base_model_whole_predictions
    else:
        all_scores = base_model_scores
        all_predictions = base_model_predictions
        all_whole_predictions = base_model_whole_predictions

    out_scores = convert_scores_to_dataframe(all_scores)
    out_predictions = all_predictions
    out_whole_predictions = all_whole_predictions
    # Save outputs
    save_results(out_scores, 'scores', args, directory)
    save_results(feature_importance_df, 'imp', args, directory)
    save_results(out_predictions, 'predictions', args, directory)
    save_results(out_whole_predictions, 'whole_predictions', args, directory)
    save_models(base_models, 'baseModel', args, directory)
    
    if perform_feature_selection:
        out_features = pd.DataFrame.from_dict(selected_features, orient='index').transpose() 
        save_results(out_features, 'features', args, directory)
    if ensemble:
        save_models(stacking_clf, 'stackingModel', args, directory)

    # Generate plots
    generate_boxplots(out_scores, args, directory)
    analyze_predictions(out_predictions, args, directory)


if __name__ == "__main__":
    
    # Define a custom type conversion function to convert comma-separated strings to lists
    def comma_separated_items(arg):
        # 分割字符串并去除每个项的前后空格
        items = [item.strip() for item in arg.split(',')]
        # 检查每个项是否都在 choices 列表中
        valid_choices = ['RandomForest', 'CatBoost', 'LogisticRegression', 'MLP', 'XGBoost', 'lasso', 'SVM', 'ElasticNet', 'NeuralNetwork', 'GaussianNB', 'KNN', 'GMWI2']
        for item in items:
            if item not in valid_choices:
                raise argparse.ArgumentTypeError(f"Invalid choice: '{item}'. Choose from {valid_choices}.")
        return items
    
    def str2bool(value):
        if isinstance(value, str):
            if value.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif value.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Load and prepare data for machine learning.")
    parser.add_argument('--prof_file', type=str, required=True, help="Path to the prof data file. row is sampleID, col is species")
    parser.add_argument('--metadata_file', type=str, required=True, help="Path to the metadata file.")
    parser.add_argument('--target', type=str, required=True, help="Target type for filtering the metadata.")
    parser.add_argument('--target_id', type=str, required=False, help="Target ID to filter the metadata.")
    parser.add_argument('--models', type=comma_separated_items, required=False, nargs='?', const=[],
                                    help="Choose from: RandomForest, CatBoost, XGBoost, LogisticRegression, MLP, lasso, SVM, ElasticNet, NeuralNetwork, GaussianNB, KNN")
    parser.add_argument('--feature', type=str2bool, required=True, help="Whether to perform feature selection (e.g., 'Yes').")
    parser.add_argument('--turner', type=str2bool, required=True, help="Whether to perform turner (e.g., 'Yes').") 
    parser.add_argument('--ensemble', type=str2bool, required=False, help="Whether to perform ensemble (e.g., 'Yes').")
    parser.add_argument('--y_cohort_group', type=str2bool, required=False, help="Whether to return cohort information")
    parser.add_argument('--y_cohort_type', type=str, required=False, default='Project', help="Type of cohort information ('Project' or 'Disease')")
    parser.add_argument('--pres', type=str2bool, required=False, help="pres profile or RA profile")
    parser.add_argument('--cpu', type=int, required=False, help="cpu")
    parser.add_argument('--repeat', type=int, required=False, help="repeats")
    parser.add_argument('--cv', type=int, required=False, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--output', type=str, required=True, help="The output directory")

    args = parser.parse_args()
    main(args)

