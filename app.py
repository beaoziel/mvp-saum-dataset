from io import BytesIO
from flask_openapi3 import OpenAPI, Info, Tag
from flask import redirect, request, send_file
import pandas as pd
from openpyxl import load_workbook

from logger import logger
from flask_cors import CORS
from schemas import *
from machinelearning import convert, prediction
from cript import cript_code

info = Info(title="APP Obesity SAUM", version="1.0.0")
app = OpenAPI(__name__, info=info)
CORS(app)

# definindo tags
home_tag = Tag(name="Documentação", description="Seleção de documentação: Swagger, Redoc ou RapiDoc")
file_tag = Tag(name="Obesity SAUM", description="Adição e download de arquivos")


@app.get('/', tags=[home_tag])
def home():
    """Redireciona para /openapi, tela que permite a escolha do estilo de documentação.
    """
    return redirect('/openapi')

@app.post('/dataset/upload', tags=[file_tag],
          responses={"200": FileSchema, "400": ErrorSchema})
def get_excel():
     """Realiza o upoload e a conversão do arquivo XLSX submetido pelo usuário
    """
     try:
        input_file = request.files['file']
        input_file.save('uploads/' + input_file.filename)
        pd.read_excel('uploads/' + input_file.filename)
        result_input = convert.convert_excel_to_csv('uploads/' + input_file.filename, input_file.filename)
        if result_input[1] == 200:
              user_result = prediction.predict_input_data(result_input[0])
              all_results = prediction.set_message(str(user_result[0]))
              return {"group": all_results[0], "group_name": all_results[1], "text": all_results[2], "status" : 200}
        else:
              error_msg = "Erro ao tratar arquivo Excel"
              logger.warning(f"Erro ao tratar arquivo Excel, {error_msg}")
              return {"message": error_msg, "status": 400}
     except Exception as e:
                error_msg = "Não foi possível salvar arquivo"
                logger.warning(f"Erro ao salvar e tratar arquivo Excel, {error_msg}")
                return {"message": "Erro ao salvar e tratar arquivo Excel", "status": 500}
     
@app.get('/dataset/download', tags=[file_tag], responses={"200": FileSchema, "400": ErrorSchema})  
def download_template():
    """Realiza o download do template
    """
    try:
        wb = load_workbook('template/template.xlsx')
        template_excel = BytesIO()
        wb.save(template_excel)
        template_excel.seek(0)
        return send_file(template_excel, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                         as_attachment=True, download_name='template.xlsx'),200
    except Exception as e:
                error_msg = "Não foi possível baixar arquivo"
                logger.warning(f"Erro ao fazer download do Excel, {error_msg}")
                return {"message": error_msg}, 400

@app.post('/developer/upload', tags=[file_tag],
          responses={"200": FileSchema, "400": ErrorSchema})
def get_new_dataset():
    """Realiza o upoload do modelo que será servido de forma embarcada no back-end
    """
    try:
        input_file = request.files['file']
        input_file.save('ObesityDataSet.csv')
        pd.read_csv('ObesityDataSet.csv')
        return {"message": "Upload do novo dataset realizado com sucesso", "status": 200}
    except Exception as e:
        error_msg = "Não foi possível o upload do novo dataset."
        logger.warning(f"Não foi possível o upload do novo dataset")
        return {"message": error_msg, "status": 400}

@app.post('/developer/code', tags=[file_tag], responses={"200": CodeSchema, "400": ErrorSchema})  
def get_code():
     """Realiza a comparação do código informado para a liberação do download
    """
     input_value = request.form.get('input_value')
     if cript_code.compare(input_value):
        return {"message": "Código informado corretamente", "status": 200}
     else:
        return {"message": "Codigo informado inválido", "status": 403}