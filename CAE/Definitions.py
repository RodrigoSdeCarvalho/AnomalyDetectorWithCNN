import os
import sys

sys.dont_write_bytecode = True

CAE_VAR_LIST = ["var1", "var2", "var3", "var_4", "var_5"]

SRC_PATH = os.getcwd()
MODELS_PATH = os.path.join(SRC_PATH, "CAE", "model")
TRAIN_DATA_PATH = os.path.join(SRC_PATH, "data", "train_data")
TEST_DATA_PATH = os.path.join(SRC_PATH, "data", "test_data", "test_data.csv")
THRESHOLDS_PATH = os.path.join(SRC_PATH, "CAE", "thresholds")
