import os
import sys

sys.path.insert(1, os.getcwd())
from machinelearning import prediction

def test_set_message() :
    assert prediction.set_message("[0.0]") == ('0', 'Abaixo do peso', "'Abaixo do peso'. Isso significa que seu IMC pode ser um valor menor que 18,5")
    assert prediction.set_message("[1.0]") == ('1', 'Peso normal', "'Peso normal'. Isso signifca que seu IMC pode estar entre 18,5 e 24,9")
    assert prediction.set_message("[2.0]") == ('2', 'Obesidade I', "'Obesidade I'. Isso significa que seu IMC pode estar entre 30 e 34,9")
    assert prediction.set_message("[3.0]") == ('3', 'Obesidade II', "'Obesidade II'. Isso significa que seu IMC pode estar entre 35 e 39,9")
    assert prediction.set_message("[4.0]") == ('4', 'Obesidade III', "'Obesidade III'. Isso significa que seu IMC pode ser superior a 40")
    assert prediction.set_message("[5.0]") == ('5', 'Sobrepeso I', "'Sobrepeso I'. Isso significa que seu IMC pode estar entre 25 e 26,9")
    assert prediction.set_message("[6.0]") == ('6', 'Sobrepeso II', "'Sobrepeso II'. Isso significa que seu IMC pode estar entre 27 e 29,9")


def test_set_message_error() :
    assert prediction.set_message("") == ({"message": "Erro ao realizar predição do modelo"}, 400)
    assert prediction.set_message("8") == ({"message": "Erro ao realizar predição do modelo"}, 400)