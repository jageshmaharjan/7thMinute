# from transformers import tokenization_bert as tokenizer
import transformers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sample_txt = "It's not raining today, in Singaprore"



df = pd.read_csv('/datadisk/BertIntentDetection/dataset/sentiment/reviews.csv')
df.info()
print(df.head())
print(df.shape)

sns.countplot(df.score)
plt.xlabel('review score')
plt.show()

def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    else:
        return 2

df['sentiment'] = df.score.apply(to_sentiment)

class_name = ['negative', 'neureal', 'positive']

ax = sns.countplot(df.sentiment).set_xticklabels(class_name)
plt.xlabel('review sentiment')
# ax.set_xticklabels(class_name)
plt.show()


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

encoding = tokenizer.encode_plus(text=sample_txt, max_length=32,
                                               add_special_tokens=True,
                                               pad_to_max_length=True,
                                               truncation=True,
                                               return_attention_mask=True,
                                               return_token_type_ids=False,
                                               #return_tensors='tf'
                                                )


print(encoding['input_ids'])
print(encoding['attention_mask'])

# token_len = []
#
# for line in df.content:
#     tokens = tokenizer.encode(line, max_length=512)
#     token_len.append(len(tokens))
#
# sns.distplot(token_len)

