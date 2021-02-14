import argparse
import os, sys

import numpy as np
import tensorflow as tf

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


def create_model(max_seq_len, bert_config_file, classes):
    with tf.io.gfile.GFile(bert_config_file, 'r') as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name='bert')

    input_ids = tf.keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name='input_ids')
    bert_output = bert(input_ids)

    print("bert_shape", bert_output.shape)

    cls_out = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = tf.keras.layers.Dropout(0.5)(cls_out)
    logits = tf.keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = tf.keras.layers.Dropout(0.5)(logits)
    logits = tf.keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = tf.keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    # load_stock_weights(bert, bert_ckpt_file)

    return model


def main(args):
    pre_trainedbert_dir = args.bert_config # '/home/jugs/Desktop/BERT-Pretrained/uncased_L-12_H-768_A-12/1'
    bert_config_file = os.path.join(pre_trainedbert_dir, 'bert_config.json')
    # bert_ckpt_file = os.path.join(pre_trainedbert_dir, 'bert_model.ckpt')
    tokenizer = FullTokenizer(vocab_file=os.path.join(pre_trainedbert_dir, 'vocab.txt'))
    max_seq_len = 128
    classes = ["negative", "neutral", "positive"]

    sentences = ["I have a horrible day", "where is my book", "it will be a great"]

    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    model = create_model(max_seq_len, bert_config_file, classes)
    model.load_weights(args.model_path) # '/datadisk/BertIntentDetection/ckpt'

    prediction = model.predict(pred_token_ids).argmax(axis=1)

    for text, label in zip(sentences, prediction):
        print("text: ", text, "\nsentiment: ", classes[label])
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument for inference the model")
    #parser.add_argument('--sentence', type=str, help="pass in input argument as sentence")
    parser.add_argument('--model_path', type=str, help="trained model path")
    parser.add_argument('--bert_config', type=str, help="config path of bert")

    args = parser.parse_args()
    main(args)