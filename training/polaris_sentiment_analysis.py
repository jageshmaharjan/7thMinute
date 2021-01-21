from collections import defaultdict

# import tensorflow as tf
import argparse
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, get_linear_schedule_with_warmup, BertTokenizer, AdamW
import numpy as np
import pandas as pd


class PolarisSubtitleData(Dataset):
    def __init__(self, subtitles, targets, tokenizer, max_len):
        self.subtitles = subtitles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.subtitles)

    def __getitem__(self, item):
        subtitles = str(self.subtitles[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            subtitles,
            add_special_tokens=True,
            max_length = self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_masks=True,
            return_tensor='pt'
        )

        return {
            'subtitle_text': subtitles,
            'input_ids': encoding['input_ids'], #.flatten(),
            'attention_mask': encoding['attention_mask'], #.flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PolarisSubtitleData(
        subtitles=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def fordward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"] #.to(device)
        attention_mask = d["attention_mask"] #.to(device)
        targets = d["targets"] #.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds==targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        return correct_predictions.double() /n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["tatgets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def main(args):
    RANDOM_SEED = 12345
    BATCH_SIZE = 16
    EPOCHS = 10
    MAX_LEN = 160
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.train_data)
    print(df.head())
    print(df.info())

    df['sentiment'] = df.score.apply(to_sentiment)

    class_names = ["negative", "neutral", "positive"]

    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # token_lens = []
    # for txt in df.content:
    #     tokens = tokenizer.encode(txt, max_length=512)
    #     token_lens.append(len(tokens))

    df_train, df_test = train_test_split(df, test_size=0.15, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    data = next(iter(train_data_loader))
    data.keys()

    model = SentimentClassifier(len(class_names))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)

    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn,
                                            optimizer, device, scheduler, len(df_train))

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn,
                                       device, len(df_val))

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        import time
        if val_acc > best_accuracy:
            model_name = "polaris_sentiment_model" + str(time.time())
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser("arguments for polaris sentiment")
    parser.add_argument('--train_data', help="train data (csv) file path", type=str)
    args = parser.parse_args()
    main(args)
