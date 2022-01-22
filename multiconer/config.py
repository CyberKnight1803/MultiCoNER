import os 
import torch 

# PATHS
PATH_BASE_MODELS = os.environ.get("PATH_BASE_MODELS", "./base_models")

# TAGS
WNUT_BIO_ID = {
    'O': 0,
    'B-CORP': 1, 
    'I-CORP': 2, 
    'B-CW': 3, 
    'I-CW': 4, 
    'B-GRP': 5, 
    'I-GRP': 6, 
    'B-LOC': 7, 
    'I-LOC': 8, 
    'B-PER': 9, 
    'I-PER': 10, 
    'B-PROD': 11, 
    'I-PROD': 12, 
}

BIO_TAGS = {
    'Other': 0,
    'B-Per': 1,
    'I-Per': 2, 
    'B-Org': 3, 
    'I-Org': 4, 
    'B-Loc': 5, 
    'I-Loc': 6
}