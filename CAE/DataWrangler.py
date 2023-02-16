import pandas as pd
import keras
import tensorflow as tf
import os
from CAE import Definitions
import sys

sys.dont_write_bytecode = True

def get_train_and_validation_datasets(df:pd.DataFrame, SEQ_LEN:int, BATCH_SIZE:int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        SEQ_LEN (int): _description_
        BATCH_SIZE (int): _description_

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: _description_
    """
    df_size = len(df)
    train_df = df[:int(0.90*df_size)]
    validation_df = df[int(0.90*df_size):int(df_size)]

    x_train, y_train = get_timeseries_dataset(train_df, SEQ_LEN), get_timeseries_dataset(train_df, SEQ_LEN)
    x_validation, y_validation = get_timeseries_dataset(validation_df, SEQ_LEN), get_timeseries_dataset(validation_df, SEQ_LEN)

    train_ds = tf.data.Dataset.zip((x_train, y_train)).batch(BATCH_SIZE).prefetch(2)
    validation_ds = tf.data.Dataset.zip((x_validation, y_validation)).batch(BATCH_SIZE).prefetch(2)

    return train_ds, validation_ds


def get_timeseries_dataset(df:pd.DataFrame, sequence_length:int) -> tf.data.Dataset:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        sequence_length (int): _description_

    Returns:
        tf.data.Dataset: _description_
    """
    timeseries_dataset = keras.utils.timeseries_dataset_from_array(
        data=tf.convert_to_tensor(df),
        targets=None,
        sequence_length=sequence_length,
        sequence_stride=1,
        batch_size=None,
    )

    return timeseries_dataset


def df_to_tensor_for_CAE_prediction(df:pd.DataFrame, sequence_length:int) -> tf.Tensor:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        sequence_length (int): _description_

    Returns:
        tf.Tensor: _description_
    """
    ds = get_timeseries_dataset(df, sequence_length)

    batched_ds = ds.batch(len(ds))

    taken_ds = batched_ds.take(1)

    iterator_ds = iter(taken_ds)

    tensor_ds = next(iterator_ds)

    return tensor_ds


def generate_train_dataframe(concat:bool = False) -> list[pd.DataFrame]:
    """_summary_

    Args:
        concat (bool, optional): _description_. Defaults to False.

    Returns:
        list[pd.DataFrame]: _description_
    """
    train_csvs_folder_path = os.path.join(os.getcwd(), Definitions.TRAIN_DATA_PATH)
    train_csvs_path_list = os.listdir(os.path.join(os.getcwd(), Definitions.TRAIN_DATA_PATH))

    train_df_list = []
    for csv in train_csvs_path_list:
        train_df_list.append(pd.read_csv(os.path.join(train_csvs_folder_path, csv), index_col=0))

    if concat:
        train_df = pd.concat(train_df_list)
        return [train_df]

    return train_df_list


def generate_test_dataframe() -> pd.DataFrame:
    """_summary_

    Args:
        concat (bool, optional): _description_. Defaults to False.

    Returns:
        list[pd.DataFrame]: _description_
    """
    return pd.read_csv(Definitions.TEST_DATA_PATH, index_col=0)     
