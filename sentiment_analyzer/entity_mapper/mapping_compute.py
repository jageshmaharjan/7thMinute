import json
import pandas as pd
from nltk import ngrams

with open("config.json") as json_file:
    config = json.load(json_file)

def create_n_grams(req_sentence, n):
    req_sentence = str(req_sentence).lower()
    i = 1
    n_gram_tokens = []
    split_sentence = req_sentence.split()
    while i < n:
        for val in split_sentence:
            tok = []
            tok.append(val)
            while i != len(tok):
                j = i+1
                tok.append(split_sentence[j])
            n_gram_tokens.append(tok)
        i += 1


def compute(request):
    whitelist = config["ER_MAPPER_CSV"]
    with open(whitelist, "r") as fp:
        data = fp.readlines()

    values = [((line.split('\t'))[0]).lower() for line in data]
    entities = [((line.split('\t'))[1]).lower().strip() for line in data]
    value_entity_map = {}

    for i, val in enumerate(values):
        if len(val.split()) <= 4:
            value_entity_map[val] = entities[i]

    all_grams = []
    unigram = ngrams(request.split(), 1)
    biggram = ngrams(request.split(), 2)
    trigram = ngrams(request.split(), 3)
    for gram in unigram:
        all_grams.append(' '.join(gram).lower())
    for gram in biggram:
        all_grams.append(' '.join(gram).lower())
    for gram in trigram:
        all_grams.append(' '.join(gram).lower())

    tokens = []
    entity = []
    for tok in all_grams:
        if tok in value_entity_map.keys():
            tokens.append(tok)
            entity.append(value_entity_map[tok])
    return tokens, entity