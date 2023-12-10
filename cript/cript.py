from hashlib import sha256

def compare(code):
    right_code = 'aca9609b87b362532d4aef6e773a8565547a4e04221371451ce52eeb5448e549'
    byte_code = bytes(code, 'UTF-8')
    hash_code = sha256(byte_code)
    
    if (hash_code.hexdigest() == right_code):
        return True
    else:
        return False

