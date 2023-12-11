import os
import sys

sys.path.insert(1, os.getcwd())
from machinelearning import prediction


no_data = {}

data_0 = {
    "Gender": "Female",
    "Age": 16,
    "Height": 1.56,
    "Weight": 36,
    "family_history_with_overweight": "no",
    "FAVC": "no",
    "FCVC": 2,
    "NCP": 1,
    "CAEC": "no",
    "SMOKE": "yes",
    "CH2O": 1,
    "SCC": "yes",
    "FAF": 3,
    "TUE": 2,
    "CALC": "no",
    "MTRANS": "Walking",
}

data_1 = {
    "Gender": "Male",
    "Age": 20,
    "Height": 1.65,
    "Weight": 56,
    "family_history_with_overweight": "yes",
    "FAVC": "no",
    "FCVC": 2,
    "NCP": 4,
    "CAEC": "no",
    "SMOKE": "no",
    "CH2O": 1,
    "SCC": "no",
    "FAF": 2,
    "TUE": 2,
    "CALC": "no",
    "MTRANS": "Automobile",
}

data_2 = {
    "Gender": "Male",
    "Age": 36,
    "Height": 1.77,
    "Weight": 92,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 2,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 1,
    "SCC": "no",
    "FAF": 2,
    "TUE": 2,
    "CALC": "no",
    "MTRANS": "Motorbike",
}

data_3 = {
    "Gender": "Female",
    "Age": 29,
    "Height": 1.63,
    "Weight": 125,
    "family_history_with_overweight": "no",
    "FAVC": "yes",
    "FCVC": 1,
    "NCP": 4,
    "CAEC": "Always",
    "SMOKE": "yes",
    "CH2O": 2,
    "SCC": "no",
    "FAF": 0,
    "TUE": 2,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation",
}


def test_predict_input_data():
    assert prediction.predict_input_data(data_0) == (([0.0]), 200) or (
        ([1.0]),
        200,
    )  # Pessoa com estilo de vida do grupo 0 ou 1 - abaixo
    assert prediction.predict_input_data(data_1) == (([1.0]), 200) or (
        ([0.0]),
        200,
    )  # Pessoa com estilo de vida do grupo 0 ou 1 - normal
    assert prediction.predict_input_data(data_2) == (([5.0]), 200) or (
        ([6.0]),
        200,
    )  # Pessoa com estilo de vida do grupo 5 ou 6 - sobrepeso
    assert (
        prediction.predict_input_data(data_3) == (([2.0]), 200)
        or (([3.0]), 200)
        or (([4.0]), 200)
    )  # Pessoa com estilo de vida do grupo 2,3 ou 4 - obesidade


def test_predict_input_data_error():
    assert prediction.predict_input_data(no_data) == (
        {"message": "Erro ao realizar predição do modelo"},
        400,
    )
