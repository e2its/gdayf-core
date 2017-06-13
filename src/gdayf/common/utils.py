from hashlib import md5 as md5
from hashlib import sha256 as sha256
from pandas import read_json
from pandas import read_json

def hash_key(hash_type, filename):
    if hash_type == 'MD5':
        try:
            return md5(open(filename, 'rb').read()).hexdigest()
        except IOError:
            raise
        return None
    elif hash_type == 'SHA256':
        try:
            return sha256(open(filename, 'rb').read()).hexdigest()
        except IOError:
            raise
        return None


def decode_json_to_dataframe(json_string, orient='split'):
    return read_json (json_string, orient=orient)


def generate_commands_parameters(each_model, model_command, train_command, train_parameters_list):
    for key, value in each_model['parameters'].items():
        if value['seleccionable']:
            if isinstance(value['value'], str):
                if key in train_parameters_list and value is not None:
                    train_command.append(", %s=\'%s\'" % (key, value['value']))
                else:
                    model_command.append(", %s=\'%s\'" % (key, value['value']))
            else:
                if key in train_parameters_list and value is not None:
                    train_command.append(", %s=%s" % (key, value['value']))
                else:
                    model_command.append(", %s=%s" % (key, value['value']))

def need_factor(atype, y, training_frame=None, valid_frame=None, predict_frame=None):
    if atype in ['binomial', 'multinomial']:
        if training_frame is not None:
            if isinstance(training_frame[y], (int, float)):
                training_frame[y] = training_frame[y].asfactor()
            else:
                training_frame[y] = training_frame[y].ascharacter().asfactor()
        if valid_frame is not None:
            if isinstance(valid_frame[y], (int, float)):
                valid_frame[y] = valid_frame[y].asfactor()
            else:
                valid_frame[y] = valid_frame[y].ascharacter().asfactor()
        if predict_frame is not None:
            if isinstance(predict_frame[y], (int, float)):
                predict_frame[y] = predict_frame[y].asfactor()
            else:
                predict_frame[y] = predict_frame[y].ascharacter().asfactor()

def get_tolerance(columns, objective_column, tolerance=0):
    min_val = None
    max_val = None
    for each_column in columns:
        if each_column["name"] == objective_column:
            min_val = float(each_column["min"])
            max_val = float(each_column["max"])
    if min_val is None or max_val is None:
        tolerance = 0
    else:
        tolerance = (max_val - min_val) * 0.005
    return tolerance