import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from os.path import exists
import CAE.Definitions as Definitions
import CAE.Model as Model
import CAE.DataWrangler as dw

sys.dont_write_bytecode = True

# Number of variables that must detect an anomaly for it to be considered an anomaly.
NUM_OF_VARS_DETECTING_AN_ANOMALY = 1

def detect() -> pd.DataFrame:
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """
    df_to_detect = dw.generate_test_dataframe()

    if df_to_detect is not None and len(df_to_detect) >= Model.SEQ_LEN:
        detections = run_anomaly_detection(df_to_detect)
    else:
        print(f"There is not enough data to detect anomalies. At least {Model.SEQ_LEN} records are needed.")
        return 

    if (detections is not None):
        ultima_data = detections.index[ len(detections.index) - 1]
        csv_file_name = f"detections{ultima_data}.csv"
        detections.to_csv(f"{Definitions.SRC_PATH}/anomalies/{csv_file_name}")
        print(f"Detections saved to {Definitions.SRC_PATH}/anomalies/{csv_file_name}")

    return detections


def run_anomaly_detection(df_to_detect:pd.DataFrame, num_of_vars_detecting_an_anomaly:int = NUM_OF_VARS_DETECTING_AN_ANOMALY) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        num_of_vars_detecting_an_anomaly (int, optional): _description_. Defaults to NUM_OF_VARS_DETECTING_AN_ANOMALY.

    Returns:
        pd.DataFrame: _description_
    """
    model = Model.load_CAE_model()

    ds = dw.df_to_tensor_for_CAE_prediction(df_to_detect, Model.SEQ_LEN)

    reconstruted_ds = model.predict(ds)

    scores = pd.DataFrame(index=df_to_detect.index)
    for var in Definitions.CAE_VAR_LIST:
        var_scores = detect_anomalies_in_an_var(var, df_to_detect, ds, reconstruted_ds)
        scores = scores.join(var_scores)

    score_column = get_score_column(scores, num_of_vars_detecting_an_anomaly)
    anomaly_column = get_anomaly_column(score_column) 

    detections = pd.DataFrame({'Anomaly': anomaly_column, 'Score': score_column}, index=df_to_detect.index)

    return detections


def detect_anomalies_in_an_var(var:str, df_to_detect:pd.DataFrame, ds:tf.Tensor, reconstruted_ds:np.ndarray) -> pd.Series:
    """_summary_

    Args:
        var (str): _description_
        df (pd.DataFrame): _description_
        ds (tf.Tensor): _description_
        reconstruted_ds (np.ndarray): _description_

    Returns:
        pd.DataFrame: _description_
    """
    var_df = df_to_detect[var]

    threshold = get_var_threshold(var)

    mae_loss = Model.calculate_var_reconstruction_mae_loss(var, ds, reconstruted_ds)

    vectorized_calculate_score = np.vectorize(calculate_score)
    score = vectorized_calculate_score(mae_loss, threshold, 2)

    scores = get_var_score_series(var, var_df, score)

    return scores


def get_var_threshold(var:str) -> float:
    """_summary_

    Args:
        var (str): _description_

    Returns:
        float: _description_
    """
    thresholds_csv_path = f"{Definitions.THRESHOLDS_PATH}/CAE_thresholds.csv"
    
    threshold_df = pd.read_csv(thresholds_csv_path, index_col=0)

    threshold_var = threshold_df.threshold.loc[var]

    return threshold_var


def calculate_score(mae_loss:float, threshold:float, ref_threshold_multiplier:float) -> float:
    """_summary_

    Args:
        mae_loss (float): _description_
        threshold (float): _description_
        ref_threshold_multiplier (float): _description_

    Returns:
        float: _description_
    """
    reference_value = threshold * ref_threshold_multiplier

    if mae_loss < threshold:
        return 0.0
    elif mae_loss > reference_value:
        return 1.0
    else:
        return abs(mae_loss / reference_value)


def get_var_score_series(var:str, var_df:pd.DataFrame, score:np.ndarray) -> pd.Series:
    """_summary_

    Args:
        var (str): _description_
        var_df (pd.DataFrame): _description_
        score (np.ndarray): _description_

    Returns:
        pd.Series: _description_
    """
    score_mask = score >= 0

    score_idxs_pos_shift = var_df.shift((Model.SEQ_LEN-1)).dropna().iloc[score_mask].index
    score_idxs_neg_shift = var_df.shift(-(Model.SEQ_LEN-1)).dropna().iloc[score_mask].index

    scores_pos_shift = pd.Series(score, name=var, index=score_idxs_pos_shift).iloc[len(score_idxs_pos_shift) - (Model.SEQ_LEN - 1):]
    scores_neg_shift = pd.Series(score, name=var, index=score_idxs_neg_shift)

    scores = pd.concat([scores_neg_shift, scores_pos_shift])

    return scores


def get_anomaly_column(score_column:pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        score_column (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    anomaly_column = score_column.apply(lambda x: 1 if x > 0 else 0)

    return anomaly_column


def get_score_column(scores:pd.DataFrame, num_of_vars_detecting_an_anomaly:int) -> pd.DataFrame:
    """_summary_

    Args:
        scores (pd.DataFrame): _description_
        num_of_vars_detecting_an_anomaly (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    score_column:pd.Series = scores.apply(combine_all_vars_scores, num_of_vars_detecting_an_anomaly = num_of_vars_detecting_an_anomaly, axis=1)

    score_column.name = "score"
    
    return score_column


def combine_all_vars_scores(row:list, num_of_vars_detecting_an_anomaly:int) -> float:
    """_summary_

    Args:
        row (list): _description_
        num_of_vars_detecting_an_anomaly (int): _description_

    Returns:
        float: _description_
    """
    num_of_vars_with_score_above_zero = len(list(filter(lambda x : x > 0, row)))

    if num_of_vars_with_score_above_zero >= num_of_vars_detecting_an_anomaly:
        return sum(row) / num_of_vars_with_score_above_zero
    else:
        return 0
