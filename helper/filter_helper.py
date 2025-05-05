import json
from helper import file_helper

def create_filter_from_json(gene, age):

    filter = file_helper.read_json( "./utils/filter.json" ) 
 
    def replace(obj):
        if isinstance(obj, str):
            return obj.replace('GENE_FAULT', gene).replace('USER_AGE', str(age))
        elif isinstance(obj, list):
            return [replace(elem) for elem in obj]
        elif isinstance(obj, dict):
            return {k: replace(v) for k, v in obj.items()}
        return obj

    return replace(filter)