import json

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForTokenClassification
import json
import numpy as np

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])

        with open(config["TAG2IDX"], 'r') as f:
            self.tag2idx = json.load(f)

        with open(config["TAG_VALUES"], 'r') as f:
            f_read = f.readlines()

        self.tag_values = []
        for val in f_read:
            self.tag_values.append(val.strip())

        model = BertForTokenClassification.from_pretrained(config["BERT_MODEL"], num_labels=len(self.tag2idx), output_attentions=False, output_hidden_states=False)
        model.load_state_dict(torch.load(config["ER_MODEL"], map_location=self.device))

        model = model.eval()
        self.model = model.to(self.device)

    def predict(self, text):
        tokenized_sentence = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokenized_sentence])

        with torch.no_grad():
            output = self.model(input_ids)

        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.tag_values[label_idx])
                new_tokens.append(token)

        return new_tokens, new_labels
        # return (
        #     dict(zip(new_tokens, new_labels)),
        # )


er_model = Model()


def get_er_model():
    return er_model
