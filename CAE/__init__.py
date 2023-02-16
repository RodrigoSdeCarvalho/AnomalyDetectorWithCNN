import sys

sys.dont_write_bytecode = True

from CAE import Model
from CAE import AnomalyDetector

def main() -> None:
    if Model.train_model():
        Model.calculate_thresholds()
        AnomalyDetector.detect()
