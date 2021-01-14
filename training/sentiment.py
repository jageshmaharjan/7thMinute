import argparse
import os, sys
import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


# class SentimentClassifier:
#     def __init__(self, n_classes):
#         super(SentimentClassifier, self).__init__()
#         self.bert = ;
#         self.drop = tf.nn.dropout(0.3)
#         self.softmax = tf.nn.softmax(dim=1)
#
#     def fordward(self, input_ids, attention_mask):
#         _, pooled_output = self.bert(input_ids=input_ids,
#                                      attention_mask=attention_mask)
#         output = self.drop(pooled_output)


class SentimentClassifier:
    DATA_COLUMN = "content"
    LABEL_COLUMN = "sentiment"

    def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.classes = classes
        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

    def _prepare(self, df):
        x = []
        y = []
        for _, row in tqdm(df.iterrows()):
            text, label = row[SentimentClassifier.DATA_COLUMN], row[SentimentClassifier.LABEL_COLUMN]
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(tokens))
            x.append(token_ids)
            y.append(self.classes.index(label))
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len-2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)


def create_model(max_seq_len, bert_ckpt_file, bert_config_file, classes):
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

    load_stock_weights(bert, bert_ckpt_file)

    return model


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def data_preprocessing(df, class_names):
    # dataframe = pd.read_csv(df)
    df_train = df[:12000]
    df_test = df[12000:]  # pd.read_csv(args.test_csv)

    # print(df_train.head())
    # print(df_train.shape)
    # print(df_train.info())
    #
    # print(df_test.head())
    # print(df_test.shape)
    # print(df_test.info())
    #
    # sns.countplot(df_train.score)
    # plt.xlabel("review score")
    # plt.show()
    #
    # sns.countplot(df_test.score)
    # plt.xlabel("review score")
    # plt.show()
    #
    df_train["sentiment"] = df_train.score.apply(to_sentiment)
    df_test["sentiment"] = df_test.score.apply(to_sentiment)
    # ax = sns.countplot(df_train.sentiment)
    # plt.xlabel("sentiment reviews")
    # plt.show()
    # ax.set_xticklabels(class_names)
    #
    # df_test["sentiment"] = df_test.score.apply(to_sentiment)
    # ax = sns.countplot(df_test.sentiment)
    # plt.xlabel("sentiment reviews")
    # plt.show()
    # ax.set_xticklabels(class_names)
    return df_train, df_test


def main(args):
    class_names = ["negative", "neutral", "positive"]

    df = pd.read_csv(args.train_csv)  # "/datadisk/BertIntentDetection/dataset/sentiment/reviews.csv"
    df_train, df_test = data_preprocessing(df, class_names)

    bert_ckpt_dir = args.bert_dir  # '/home/jugs/Desktop/BERT-Pretrained/uncased_L-12_H-768_A-12/1'
    bert_ckpt_file = os.path.join(bert_ckpt_dir, 'bert_model.ckpt')
    bert_config_file = os.path.join(bert_ckpt_dir, 'bert_config.json')
    bert_vocab_file = os.path.join(bert_ckpt_dir, 'vocab.txt')

    tokenizer = FullTokenizer(vocab_file=bert_vocab_file)
    tokens = tokenizer.tokenize("I'd like to go for a jogging, in the evening!")

    print(tokens)
    token_id = tokenizer.convert_tokens_to_ids(tokens)
    print(token_id)

    classes = df_train.sentiment.unique().tolist()
    print(classes)

    data = SentimentClassifier(df_train, df_test, tokenizer, classes, max_seq_len=128)

    print(data.train_x.shape)

    print(data.train_x[0])
    print(data.train_y[0])
    print(data.max_seq_len)

    model = create_model(data.max_seq_len, bert_ckpt_file, bert_config_file, classes)

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
    )

    log_dir = 'log/sentiment/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ckpt_file = '/datadisk/BertIntentDetection/sentiment_model'
    model_ckpts = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_file,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True
    )

    history = model.fit(x=data.train_x, y=data.train_y,
                        validation_split=0.1,
                        batch_size=16,
                        shuffle=True,
                        epochs=5,
                        callbacks=[tensorboard_callback, model_ckpts])

    _, train_acc = model.evaluate(data.train_x, data.train_y)
    _, test_acc = model.evaluate(data.test_x, data.test_y)

    print("train acc", train_acc)
    print("test acc", test_acc)

    y_pred = model.predict(data.test_x).argmax(axis=-1)

    print(classification_report(data.test_y, y_pred, target_names=classes))

    cm = confusion_matrix(data.test_y, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for training the Sentiment Classifier using BERT")
    parser.add_argument('--train_csv', type=str, help="training data file")  # "/datadisk/BertIntentDetection/dataset/sentiment/reviews.csv"
    # parser.add_argument('--test_csv', type=str, help="training data file")   # "/datadisk/BertIntentDetection/dataset/sentiment/apps.csv"
    parser.add_argument('--bert_dir', type=str, help="file path to BERT dir")

    args = parser.parse_args()
    main(args)