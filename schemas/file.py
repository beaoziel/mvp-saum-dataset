from pydantic import BaseModel


class FileSchema(BaseModel):
    """ Define como uma mensagem de erro será representada
    """
    message: str

class CodeSchema(BaseModel):
  """ Define como um codigo deve ser representado
    """
  code: str