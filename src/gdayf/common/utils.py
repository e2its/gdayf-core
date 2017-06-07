from os import makedirs


def mkdir(path, grants):
    try:
        if not path.exists(path):
            makedirs(path, grants)
        return 0
    except IOError:
        return 1
