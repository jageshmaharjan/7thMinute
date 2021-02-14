import json
import os


def get_config():
    with open('config.json', 'r') as cfg:
        config = json.load(cfg)
    return config


def view_models():
    res = os.listdir('assets/')
    return [bin for bin in res if bin.split('.')[-1] == 'bin']