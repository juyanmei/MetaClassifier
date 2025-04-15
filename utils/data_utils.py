import pandas as pd
import numpy as np
import logging
import sys
import datetime
import warnings
import os
import atexit
from joblib import dump, load

def load_and_prepare_data(prof_file, metadata_file, target):
    """
    Load data files and prepare data.

    Parameters:
    prof_file (str): Path to feature data file.
    metadata_file (str): Path to metadata file.

    Returns:
    prof (pd.DataFrame): Feature data.
    metadata (pd.DataFrame): Metadata.
    cohorts: List of unique cohorts.
    """
    # Read data files
    prof = pd.read_table(prof_file, sep=',', index_col=0)
    metadata = pd.read_table(metadata_file, sep=',', index_col=0)
    cohorts = pd.unique(metadata[target].values).tolist()
    print("----------------------------")
    print("Data loading...")
    print(f"    The original prof is: {prof.shape}")
    print(f"    The original metadata is: {metadata.shape}")
    print()
    print("----------------------------")
    return prof, metadata, cohorts

def filter_data(prof, metadata, target, target_id = None, y_cohort_group = False, y_cohort_type = 'Project', pres = True):
    """
    Filter data based on target column and target value.

    Parameters:
    prof (pd.DataFrame): Feature data.
    metadata (pd.DataFrame): Metadata.
    target (str): Target column name.
    target_id (str): Target value.
    y_cohort_group (bool): Whether to return cohort information.
    y_cohort_type (str): Type of cohort information ('Project' or 'Disease').

    Returns:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target vector.
    """

    print("Data filtering...")
    
    if target_id == 'None':
        target_id = None
    
    if target_id is not None:
        # Filter metadata based on target
        if target == 'Project':
            metadata = metadata[metadata['Project'] == target_id]
            print(f"    For Project {target_id}, the filtered metadata is: {metadata.shape}")
        elif target == 'Disease':
            metadata = metadata[metadata['Disease'] == target_id]
            print(f"    For Disease {target_id}, the filtered metadata is: {metadata.shape}")
        elif target == 'SubMajorDisease':
            metadata = metadata[metadata['SubMajorDisease'] == target_id]
            print(f"    For Disease {target_id}, the filtered metadata is: {metadata.shape}")
        else:
            raise ValueError("Invalid target value. It should be either 'Project' or 'Disease'.")

    filtered_prof = prof[prof.index.isin(metadata.index)]
    Group = np.array([1 if i == "Disease" else 0 for i in metadata["Group"]])
    
    if y_cohort_group:
        if y_cohort_type not in ['Project', 'Disease']:
            raise ValueError("y_cohort_type must be either 'Project' or 'Disease'")
        y_cohort = np.array(metadata[y_cohort_type])
    else:
        y_cohort = None
    
    # Prepare feature matrix and target vector
    if pres:
        X = filtered_prof.applymap(lambda x: 1 if x > 1e-4 else 0)
    else:
        X = filtered_prof
    y = Group
    print(f"The filtered prof is: {X.shape}")
    print()
    print("----------------------------")
    return X, y, y_cohort



def convert_scores_to_dataframe(all_scores):
    """
    Convert model scoring dictionary to DataFrame.

    Parameters:
    - all_scores (dict): Dictionary containing model scores.

    Returns:
    - pd.DataFrame: Converted DataFrame.
    """
    rows = []
    for model_name, scores in all_scores.items():
        for i in range(len(scores['test_accuracy'])):
            row = {
                'Model': model_name,
                'Accuracy': scores['test_accuracy'][i],
                'Precision': scores['test_precision'][i],
                'Recall': scores['test_recall'][i],
                'F1': scores['test_f1'][i],
                'ROC_AUC': scores['test_roc_auc'][i],
                'MCC': scores['test_mcc'][i],
                'Specificity': scores['test_specificity'][i],
                'CV': scores['CV'][i] if 'CV' in scores else 'NA',
                'Repeat': scores['Repeat'][i] if 'Repeat' in scores else -1,
            }
            rows.append(row)
    return pd.DataFrame(rows)
    
def remove_duplicate_columns(df):
    """
    Remove duplicate columns from the dataframe
    :param df: Input dataframe
    :return: Dataframe with duplicate columns removed
    """
    # Check for duplicate column names
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        # Get indices of duplicate columns
        dup_indices = df.columns.get_loc(dup)
        # Keep the first occurrence of the column, remove subsequent duplicates
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def setup_logging(log_dir='.'):
    """
    Set up logging system, ensuring log file paths are correct
    
    Parameters:
        log_dir (str): Directory for storing log files
    
    Returns:
        logging.Logger: Configured logger
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Build log file paths with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'output_{timestamp}.log')
    warning_file = os.path.join(log_dir, f'warning_{timestamp}.log')
    
    # Main log configuration - for handling regular logs and print output
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    
    # Create dedicated warning log handler
    warning_logger = logging.getLogger('warning_logger')
    warning_logger.setLevel(logging.WARNING)  # Only handle WARNING level and above
    warning_logger.propagate = False  # Prevent propagation to other handlers
    
    # Add warning log file handler
    warning_handler = logging.FileHandler(warning_file, mode='a', encoding='utf-8')
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    warning_logger.addHandler(warning_handler)
    
    # Custom warning handling function
    def log_warnings_to_file(message, category, filename, lineno, file=None, line=None):
        warning_logger.warning(f"{filename}:{lineno} - {category.__name__}: {message}")
        for handler in warning_logger.handlers:
            handler.flush()
    
    # Redirect warnings to custom handling function
    warnings.showwarning = log_warnings_to_file
    
    # Redirect standard output to log file
    sys.stdout = open(log_file, 'a')
    atexit.register(lambda: sys.stdout.close())
    # Ensure file handlers are properly closed
    for handler in logger.handlers + warning_logger.handlers:
        handler.close()

    return logger


def save_results(df, df_name, args, directory):
    df['target'] = args.target
    tmp_target_id = getattr(args, 'target_id', 'merge') 
    df['target_id'] = tmp_target_id
    df['feature'] = args.feature
    df['turner'] = args.turner
    df_filename = f"{directory}/{args.target}_{tmp_target_id}_{df_name}_Feature{args.feature}_Turner{args.turner}.csv"
    with open(df_filename, 'w') as f:
        df.to_csv(f, index=True)

def save_models(model, name, args, directory):
    tmp_target_id = getattr(args, 'target_id', 'merge')
    model_filename = f"{directory}/{args.target}_{tmp_target_id}_{name}_Feature{args.feature}_Turner{args.turner}.joblib"
    with open(model_filename, 'wb') as f:
        dump(model, f)