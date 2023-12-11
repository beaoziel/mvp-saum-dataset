import os
import sys

sys.path.insert(1, os.getcwd())
from machinelearning import convert


def test_convert_excel_to_csv():
    assert convert.convert_excel_to_csv(
        "C:/Users/biaoz/OneDrive/Documentos/mvp-saum-dataset/uploads/template (5).xlsx",
        "template (5).xlsx",
    ) == (
        {
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
        },
        200,
    )


def test_convert_excel_to_csv_error():
    assert convert.convert_excel_to_csv("/uploads/oscars.csv", "oscars.csv") == (
        {"message": "Erro ao realizar conversão do arquivo"},
        400,
    )
