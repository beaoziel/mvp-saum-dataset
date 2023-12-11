import os
import sys

sys.path.insert(1, os.getcwd())
from cript import cript_code


def test_compare():
    assert cript_code.compare("sou_saum_dev!!") == True


def test_compare_error():
    assert cript_code.compare("teste_senha") == False
