import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertForTokenClassification
import numpy as np

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CLASS_NAMES = ["negative", "neutral", "positive"]

    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    model = SentimentClassifier(len(CLASS_NAMES))
    model.load_state_dict(torch.load("/datadisk/7thMinute/assets/model_state_dict.bin",
                                                           map_location=torch.device("cpu")))

    model = model.eval()

    test_sentence = "i am very exhausted and tired"

    tokenized_sentence = tokenizer.encode_plus(test_sentence)
    input_ids = torch.tensor([tokenized_sentence])

    with torch.no_grad():
        output = model(input_ids)

    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []

    print()