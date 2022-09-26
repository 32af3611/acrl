import json
import os
import pickle


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def write_json(file, obj, **kwargs):
    with open(file, "w") as f:
        return json.dump(obj, f, **kwargs)


def read_pickle(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(file, data):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
