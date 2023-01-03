import json
import jsonpickle


def encode_indent(object):
    """
    Method to encode JSON

    Parameters
    ----------
    object : object

    Returns
    -------
    JSONEncoder
        Encoded JSON object

    """
    return json.dumps(json.loads(jsonpickle.encode(object)), indent=4, sort_keys=True)


def save_any_json(dict2save: dict, path: str):
    frozen = jsonpickle.encode(dict2save)
    with open(path, 'w') as fp:
        json.dump(frozen, fp)


def load_any_json(path: str):
    with open(path, 'r') as fp:
        read_dict = json.load(fp)

    return jsonpickle.decode(read_dict)
