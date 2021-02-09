import os
import sys

import argparse
import pandas as pd
import json
import numpy as np
import requests


def get_sentence_sentiment(query):
    endpoints = "http://project.delvify.ai:8000/sentiment"
    headers = {"content-type":"application-json"}
    data = json.dumps({"text":query})
    response = requests.post(endpoints, data=data, headers=headers)
    prediction = json.loads(response.text)['sentiment']
    return {"query": query, "pred":prediction}


# with open("valid.csv", 'w') as f:
#     for res in results:
#         f.write(res + '\n')
# f.close()

def main(args):
    with open(args.valid_data, "r") as f:
        sentences = f.readlines()

    results = []
    for sent in sentences:
        res = get_sentence_sentiment(sent)
        results.append(res['pred'])
        print(res['pred'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Argument for testing validation")
    parser.add_argument("--valid_data", type=str, help="validation data")
    args = parser.parse_args()
    main(args)