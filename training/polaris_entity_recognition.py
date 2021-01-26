import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import BertTokenizer, BertForTokenClassification, BertConfig, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class SentenceGetter:
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w,p,t) for w,p,t in zip(s['Word'].values.tolist(),
                                                       s['POS'].values.tolist(),
                                                       s['Tag'].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def toknize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels


def predict(tokenizer, model, tag_values):
    test_sent = "President Biden will be inagurating this noon in Pensylvenia avenue, WA at Capitol Hill"
    tokenized_sent = tokenizer.encode(test_sent)

    if torch.cuda.is_available():
        input_ids = torch.tensor([tokenized_sent]).cuda()
    else:
        input_ids = torch.tensor([tokenized_sent])

    with torch.no_grad():
        output = model(input_ids)

    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    for token, lable in zip(new_tokens, new_labels):
        print("{}\t{}".format(token, lable))


def main(args):
    data = pd.read_csv('/home/ubuntu/7thmin/webservice/resources/ner_dataset.csv',
                       encoding='latin1').fillna(method='ffill')

    print(data.head())

    getter = SentenceGetter(data)
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    print(sentences[0])
    pos = [[word[2] for word in sentence] for sentence in getter.sentences]
    print(pos[0])
    labels = [[word[2] for word in sentence] for sentence in getter.sentences]
    print(labels[0])

    tag_values = list(set(data['Tag'].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    with open("tag_values.txt", 'w+') as f_tag_val:
        for val in tag_values:
            f_tag_val.write(val + '\n')
    f_tag_val.close()

    with open("tag2idx.json", 'w+') as f_taf2idx:
        json.dump(tag2idx, f_taf2idx)
    f_taf2idx.close()

    MAX_LEN = 92
    BATCH_SIZE = 32
    PRE_TRAINED_MODEL = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL, do_lower_case=False)

    tokenized_texts_and_labels = [toknize_and_preserve_labels(tokenizer, sent, lbls)
                                  for sent, lbls in zip(sentences, labels)]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], dtype="long",
                         padding="post", truncating="post"
                         )

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    train_inputs, val_inputs, train_tags, val_tags = train_test_split(input_ids, tags, random_state=2018,
                                                                      test_size=0.1)
    train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

    train_inputs = torch.tensor(train_inputs)
    val_inputs = torch.tensor(val_inputs)
    train_tags = torch.tensor(train_tags)
    val_tags = torch.tensor(val_tags)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

    model = BertForTokenClassification.from_pretrained(PRE_TRAINED_MODEL,
                                                       num_labels=len(tag2idx),
                                                       output_attentions=False,
                                                       output_hidden_states=False)

    if torch.cuda.is_available():
        model.cuda()

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for m, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

    epochs = 20
    max_grad_norm = 1.0

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=total_steps)

    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc='Epoch'):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(),
                                          max_norm=max_grad_norm)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Avg. train loss: {}".format(avg_train_loss))

        loss_values.append(avg_train_loss)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Valid loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                      for l_i in l if tag_values[l_i] != "PAD"]

        print("Valid Acc: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Valid Acc: {}".format(f1_score(pred_tags, valid_tags)))

    import time
    model_name = "polaris_ER_model" + str(time.time()) + ".bin"
    torch.save(model.state_dict(), model_name)

    predict(tokenizer, model, tag_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for Entity Recognition")
    parser.add_argument("--train_file", type=str, help="Training file")
    args = parser.parse_args()
    main(args)
