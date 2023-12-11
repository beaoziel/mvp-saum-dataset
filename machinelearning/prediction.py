# configuração para não exibir os warnings
import warnings
import os
import re


warnings.filterwarnings("ignore")

# Imports necessários
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

results = {
    0: "Abaixo do peso",
    1: "Peso normal",
    2: "Obesidade I",
    3: "Obesidade II",
    4: "Obesidade III",
    5: "Sobrepeso I",
    6: "Sobrepeso II",
}

text_results = {
    "Abaixo do peso": f"'Abaixo do peso'. Isso significa que seu IMC pode ser um valor menor que 18,5",
    "Peso normal": f"'Peso normal'. Isso signifca que seu IMC pode estar entre 18,5 e 24,9",
    "Obesidade I": f"'Obesidade I'. Isso significa que seu IMC pode estar entre 30 e 34,9",
    "Obesidade II": f"'Obesidade II'. Isso significa que seu IMC pode estar entre 35 e 39,9",
    "Obesidade III": f"'Obesidade III'. Isso significa que seu IMC pode ser superior a 40",
    "Sobrepeso I": f"'Sobrepeso I'. Isso significa que seu IMC pode estar entre 25 e 26,9",
    "Sobrepeso II": f"'Sobrepeso II'. Isso significa que seu IMC pode estar entre 27 e 29,9",
}


def predict_input_data(data : dict):
    input_dataset = {}
    for key, value in data.items():
        v = []
        v.append(value)
        input_dataset[key] = v

    try:
        le = LabelEncoder()
        test_size = 0.20
        seed = 7
        url = "ObesityDataSet.csv"
        df = pd.read_csv(url, delimiter=",")

        # changing text values to numbers
        df["Gender"] = le.fit_transform(df["Gender"])
        df["family_history_with_overweight"] = le.fit_transform(
            df["family_history_with_overweight"]
        )
        df["FAVC"] = le.fit_transform(df["FAVC"])
        df["CAEC"] = le.fit_transform(df["CAEC"])
        df["SMOKE"] = le.fit_transform(df["SMOKE"])
        df["SCC"] = le.fit_transform(df["SCC"])
        df["CALC"] = le.fit_transform(df["CALC"])
        df["MTRANS"] = le.fit_transform(df["MTRANS"])
        df["NObeyesdad"] = le.fit_transform(df["NObeyesdad"])

        array = df.values
        X = array[:, 0:16]
        y = array[:, 16]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=seed, stratify=y
        )

        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        model = DecisionTreeClassifier(
            ccp_alpha=0.001,
            criterion="entropy",
            max_depth=9,
            max_features="sqrt",
            min_samples_leaf=2,
            min_samples_split=5,
        )

        model.fit(rescaledX, y_train)
        rescaledTestX = scaler.transform(X_test)
        predictions = model.predict(rescaledTestX)

        regressor = RandomForestRegressor()
        regressor.fit(X_train, y_train)

        # changing text values to numbers

        input_dataset["Gender"] = le.fit_transform(input_dataset["Gender"])
        input_dataset["family_history_with_overweight"] = le.fit_transform(
            input_dataset["family_history_with_overweight"]
        )
        input_dataset["FAVC"] = le.fit_transform(input_dataset["FAVC"])
        input_dataset["CAEC"] = le.fit_transform(input_dataset["CAEC"])
        input_dataset["SMOKE"] = le.fit_transform(input_dataset["SMOKE"])
        input_dataset["SCC"] = le.fit_transform(input_dataset["SCC"])
        input_dataset["CALC"] = le.fit_transform(input_dataset["CALC"])
        input_dataset["MTRANS"] = le.fit_transform(input_dataset["MTRANS"])

        atributtes = [
            "Gender",
            "Age",
            "Height",
            "Weight",
            "family_history_with_overweight",
            "FAVC",
            "FCVC",
            "NCP",
            "CAEC",
            "SMOKE",
            "CH2O",
            "SCC",
            "FAF",
            "TUE",
            "CALC",
            "MTRANS",
        ]

        input = pd.DataFrame(input_dataset, columns=atributtes)
        regressor.score(X_test, y_test)

        array_input = input.values
        X_input = array_input[:, 0:16].astype(float)
        rescaledEntradaX = scaler.transform(X_input)
        result = model.predict(rescaledEntradaX)
        return result, 200

    except Exception as e:
        error_msg = "Erro ao realizar predição do modelo"
        return {"message": error_msg}, 400


def set_message(value : str):
    try:
        number = int(value[1])
        output_title: str = ""
        output_text: str = ""

        for key, value in results.items():
            if number == key:
                output_title = value

        for key, value in text_results.items():
            if output_title == key:
                output_text = value

        return str(number), output_title, output_text
    except Exception as e:
        error_msg = "Erro ao realizar predição do modelo"
        return {"message": error_msg}, 400