import pandas as pd
import numpy as np
from keras.layers import Input, Dropout,Conv1D,Conv1DTranspose
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
import tensorflow as tf
import glob
import os, re, sys  
from os.path import exists
from CAE import Definitions
import CAE.DataWrangler as dw

sys.dont_write_bytecode = True

SEQ_LEN = 18

BATCH_SIZE = 32

EPOCHS = 60

def load_CAE_model() -> Sequential:
    """_summary_

    Returns:
        Sequential: _description_
    """
    model_file_name = f"CAE.h5"
    model_path = f"{Definitions.MODELS_PATH}/{model_file_name}"

    if exists(model_path):
        model = load_model(model_path)
        return model
    else:
        print(f"Model {model_file_name} not found in {model_path}.")


def save_CAE_model(model:Sequential) -> None:
    """_summary_

    Args:
        model (Sequential): _description_
    """
    model_file_name = f"CAE.h5"
    model_path = f"{Definitions.MODELS_PATH}/{model_file_name}"

    model.save(model_path)


def train_model() -> bool:
    """_summary_

    Returns:
        bool: Flag that indicates if the model was or had already been trained successfully.
    """
    if not has_training_data():
        print("Files with training data not found in ",f"{Definitions.TRAIN_DATA_PATH}/_data_*.csv")
        return False

    if exists(f"{Definitions.MODELS_PATH}/CAE.h5"):
        print("Model CAE.h5 already trained. Remove the file to train again.")
        return True

    train_df_list = dw.generate_train_dataframe()

    trained_flag = train_CAE(train_df_list)
    
    if trained_flag:
        print("Model CAE.h5 trained successfully.")

    return trained_flag


def has_training_data() -> bool:
    """_summary_

    Returns:
        bool: _description_
    """
    arqs = glob.glob(f'{Definitions.TRAIN_DATA_PATH}/*.csv')

    if (arqs is None) or (len(arqs) == 0):
        return False
    else:
        return True


def train_CAE(train_df_list:list) -> bool:
    """_summary_

    Args:
        df_list (list): _description_

    Returns:
        bool: _description_
    """
    model = compile_CAE_model() 

    model_weights = None

    trained_flag = False

    for train_df in train_df_list:
        if train_df is not None and len(train_df) >= SEQ_LEN:
            train_ds, validation_ds = dw.get_train_and_validation_datasets(train_df, SEQ_LEN, BATCH_SIZE)

            model = fit(model, train_ds, validation_ds, model_weights)

            model_weights = model.get_weights()

            trained_flag = True
        else:
            print(f"File with training data from {train_df.index[0]} to {train_df.index[-1]} has less than {SEQ_LEN} records, and cannot be used for training.")

    # Salva o modelo do CAE.
    if trained_flag:
        save_CAE_model(model)
        return True
    else:
        return False


def compile_CAE_model() -> Sequential:
    """_summary_

    Returns:
        Sequential: _description_
    """
    # Define os parâmetros do modelo.
    k_size = 3
    strides = 1

    # Instancia um modelo sequencial do Keras.
    model = Sequential()

    # Define o número de variáveis de entrada.
    input_vars_num = len(Definitions.CAE_VAR_LIST)

    # Adiciona a camada de input do modelo. Define o tamanho de sequência.
    model.add(Input(shape=(SEQ_LEN,input_vars_num)))

    # Adiciona as camadas de codificação ao modelo.
    model = add_encoder_layers(model, k_size, strides)

    # Adiciona as camadas de decodificação ao modelo.
    model = add_decoder_layers(model, k_size, strides)

    # Adiciona a camada de saída do modelo. Define o número de variáveis de saída.
    model.add(Conv1DTranspose(filters=input_vars_num, kernel_size=7, padding="same"))

    # Imprime o resumo das camadas do modelo.
    model.summary()

    # Compila o modelo. Define o otimizador Adam e a função de perda MSE (Mean Squared Error).
    model.compile(optimizer="adam", loss="mse" )

    return model


def add_encoder_layers(model: Sequential, k_size:int, strides:int) -> Sequential:
    """_summary_

    Args:
        model (Sequential): _description_
        k_size (int): _description_
        strides (int): _description_

    Returns:
        Sequential: _description_
    """
    # Adds the encoder layers, with 1D convolutions (Because it is a sequence, not an image). 
    # The filters are defined in a descending way, to perform the dimensionality reduction. 
    # The activation function is the ReLU (Rectified Linear Unit activation function). 
    # The Dropout layer randomly sets input units to 0, in order to avoid overfitting.
    model.add(Conv1D(filters=32, kernel_size=k_size, padding="same", strides=strides, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Conv1D(filters=16, kernel_size=k_size, padding="same", strides=strides, activation="relu"))
    model.add(Conv1D(filters=8, kernel_size=k_size, padding="same", strides=strides, activation="relu"))
    
    return model


def add_decoder_layers(model: Sequential, k_size:int, strides:int) -> Sequential:
    """_summary_

    Args:
        model (Sequential): _description_
        k_size (int): _description_
        strides (int): _description_

    Returns:
        Sequential: _description_
    """
    # Adds the decoder layers, with 1D transposed convolutions 
    # (Transpose because the transformation goes in the opposite direction to the convolution layer above).
    # The filters are defined in a ascending way, to perform the dimensionality recovery.
    # The activation function is the ReLU (Rectified Linear Unit activation function).
    model.add(Conv1DTranspose(filters=8, kernel_size=k_size, padding="same", strides=strides, activation="relu"))
    model.add(Conv1DTranspose(filters=16, kernel_size=k_size, padding="same", strides=strides, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Conv1DTranspose(filters=32, kernel_size=k_size, padding="same", strides=strides, activation="relu"))

    return model


def fit(model:Sequential, train_ds:pd.DataFrame,validation_ds:pd.DataFrame, previous_iter_weights:list) -> Sequential:
    """_summary_

    Args:
        model (Sequential): _description_
        train_ds (pd.DataFrame): _description_
        validation_ds (pd.DataFrame): _description_
        previous_iter_weights (list): _description_

    Returns:
        Sequential: _description_
    """
    es = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    if previous_iter_weights != None:
        model.set_weights(previous_iter_weights)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
    )

    return model


def calculate_thresholds() -> None:
    """_summary_

    """
    if not has_training_data():
        print("Files with data for MAE calculation not found in ",f"{Definitions.TRAIN_DATA_PATH}/_*.csv")
        return

    model = load_CAE_model()

    train_df = dw.generate_train_dataframe(True)[0]
    if train_df is None or len(train_df) < SEQ_LEN:
        print(f"Training file with data from {train_df.index[0]} to {train_df.index[-1]} contains less than {SEQ_LEN} records, and cannot be used for threshold calculation.")
        return

    original_dataset = dw.df_to_tensor_for_CAE_prediction(train_df, SEQ_LEN)
    reconstruted_dataset = model.predict(original_dataset)

    threshold_var_list = []
    for var in Definitions.CAE_VAR_LIST:
        threshold = calculate_var_threshold(var, original_dataset, reconstruted_dataset)
        threshold_var_list.append(threshold)

    save_thresholds(threshold_var_list)


def calculate_var_threshold(var:str, original_dataset:tf.Tensor, reconstruted_dataset:np.ndarray, percentil:float = 0.99) -> float:
    """_summary_

    Args:
        var (str): _description_
        original_dataset (tf.Tensor): _description_
        reconstruted_dataset (np.ndarray): _description_
        percentil (float, optional): _description_. Defaults to 0.99.

    Returns:
        float: _description_
    """
    train_mae_loss = calculate_var_reconstruction_mae_loss(var, original_dataset, reconstruted_dataset)

    threshold = np.quantile(train_mae_loss, percentil)

    return threshold


def calculate_var_reconstruction_mae_loss(var:str, ds:tf.Tensor, reconstruted_ds:np.ndarray) -> float:
    """_summary_

    Args:
        var (str): _description_
        ds (tf.Tensor): _description_
        reconstruted_ds (np.ndarray): _description_

    Returns:
        float: _description_
    """
    var_index = Definitions.CAE_VAR_LIST.index(var)
    mae_loss = np.mean(np.abs(reconstruted_ds[:,:,var_index] - ds[:,:,var_index]), axis=1)

    return mae_loss


def save_thresholds(threshold_var_list:list) -> None:
    """_summary_

    Args:
        threshold_var_list (list): _description_
    """
    vars = Definitions.CAE_VAR_LIST
    thresholds = threshold_var_list

    thresholds_df = pd.DataFrame(thresholds, index=vars, columns=['threshold'])
    thresholds_file_path = f"{Definitions.THRESHOLDS_PATH}/CAE_thresholds.csv"
    thresholds_df.to_csv(thresholds_file_path)
