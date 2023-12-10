import pandas as pd
import os

def convertGender(input):
    if input == "M":
        return "Male"
    else:
        return "Female"


def convertYesOrNo(input):
    if input == "Sim":
        return "yes"
    else:
        return "no"


def convertConditions(input):
    if input == "As vezes":
        return "Sometimes"
    elif input == "Frequentemente":
        return "Frequently"
    elif input == "Sempre":
        return "Always"
    else:
        return "no"


def convertTUE(input):
    if input == "0 - Somente o necessário":
        return 0
    elif input == "1 - Bastante":
        return 1
    else:
        return 2


def convertFAF(input):
    if input == "1 - Baixa":
        return 1
    elif input == "2 - Media":
        return 2
    elif input == "3 - Alta":
        return 3
    else:
        return 0


def convertFrequency(input):
    if input == "1 - Baixa":
        return 1
    elif input == "2 - Media":
        return 2
    elif input == "3 - Alta":
        return 3


def convertMeals(input):
    if input == "4 ou mais":
        return 4
    else:
        return input


def convertMTRANS(input):
    if input == "Moto":
        return "Motorbike"
    elif input == "Transporte publico":
        return "Public_Transportation"
    elif input == "Andar":
        return "Walking"
    elif input == "Carro":
        return "Automobile"
    else:
        return "Bike"


def convertExcelToCsv(file_path, file_name):
    try:
        excel_file_path = file_path
        df = pd.read_excel(excel_file_path)
        df.to_csv(f"output{file_name}.csv", index=False)
        df_csv = pd.read_csv(f"output{file_name}.csv")

        input_treated = {
            "Gender": convertGender(df_csv.iat[0, 0]),
            "Age": df_csv.iat[0, 1],
            "Height": df_csv.iat[0, 2],
            "Weight": df_csv.iat[0, 3],
            "family_history_with_overweight": convertYesOrNo(df_csv.iat[0, 4]),
            "FAVC": convertYesOrNo(df_csv.iat[0, 5]),
            "FCVC": convertFrequency(df_csv.iat[0, 6]),
            "NCP": convertMeals(df_csv.iat[0, 7]),
            "CAEC": convertConditions(df_csv.iat[0, 8]),
            "SMOKE": convertYesOrNo(df_csv.iat[0, 9]),
            "CH2O": convertFrequency(df_csv.iat[0, 10]),
            "SCC": convertYesOrNo(df_csv.iat[0, 11]),
            "FAF": convertFAF(df_csv.iat[0, 12]),
            "TUE": convertTUE(df_csv.iat[0, 13]),
            "CALC": convertConditions(df_csv.iat[0, 14]),
            "MTRANS": convertMTRANS(df_csv.iat[0, 15]),
        }
        
        os.remove(os.path.join(os.getcwd(), f"output{file_name}.csv"))

        return input_treated, 200
    except Exception as e:
            error_msg = "Erro ao realizar conversão do arquivo"
            return {"message": error_msg}, 400


