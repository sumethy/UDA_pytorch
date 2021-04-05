"""
prepare data in the same format as /home/users/v-sumeth/AIS/UDA_pytorch/data/*
"""

# %%
import pandas as pd
from transformers import RobertaTokenizer
import torch
import re

from fairseq.models.transformer import TransformerModel
from mosestokenizer import MosesTokenizer, MosesDetokenizer
from pythainlp.tokenize import word_tokenize as th_word_tokenize
from functools import partial
from tqdm import tqdm
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# %%
en_word_tokenize = MosesTokenizer('en')
en_word_detokenize = MosesDetokenizer('en')

en2th_word2bpe = TransformerModel.from_pretrained(
    model_name_or_path='/home/users/v-sumeth/AIS/Vistech_SCB_mt_models/SCB_1M+TBASE_en-th_moses-newmm_space_130000-130000_v1.0/models/',
    checkpoint_file='checkpoint.pt',
    data_name_or_path='/home/users/v-sumeth/AIS/Vistech_SCB_mt_models/SCB_1M+TBASE_en-th_moses-newmm_space_130000-130000_v1.0/vocab/'
)

th_word_tokenize = partial(th_word_tokenize, keep_whitespace=False)

th2en = TransformerModel.from_pretrained(
    model_name_or_path="/home/users/v-sumeth/AIS/Vistech_SCB_mt_models/SCB_1M+TBASE_th-en_newmm-moses_130000-130000_v1.0/models",
    checkpoint_file='checkpoint.pt',
    data_name_or_path="/home/users/v-sumeth/AIS/Vistech_SCB_mt_models/SCB_1M+TBASE_th-en_newmm-moses_130000-130000_v1.0/vocab",
)

en2th_word2bpe.to(torch.device('cuda'))
th2en.to(torch.device('cuda'))

# %%
def backtranslate(en_text:str) -> str:
    tokenized_sentence = ' '.join(en_word_tokenize(en_text))
    hypothesis = en2th_word2bpe.translate(tokenized_sentence)
    hypothesis = hypothesis.replace(' ', '').replace('▁', ' ').strip()
    tokenized_sentence = ' '.join(th_word_tokenize(hypothesis))
    _hypothesis = th2en.translate(tokenized_sentence)
    hypothesis = en_word_detokenize([_hypothesis])
    return hypothesis

# %%
df = pd.read_csv("/home/users/v-sumeth/AIS/UDA_pytorch/data_imdb_raw/IMDB_dataset.csv")

# %%
# clean HTML tags
reviews = []
sentiments = []
for _, row in df.iterrows():
    raw_text = row["review"]
    clean_text = re.sub("<br />", "", raw_text)
    if len(clean_text) > 250:
        clean_text = clean_text[0:250]
    reviews.append(clean_text)
    sentiments.append(row["sentiment"])

df_clean = pd.DataFrame({
    "review": reviews,
    "sentiment": sentiments
})
# %%
# data augmentation (back translation)
# this cell takes 2 days
aug_reviews = []
sentiments = []
for _, row in tqdm(df_clean.iterrows()):
    ori_text = row["review"]
    aug_text = backtranslate(ori_text)
    aug_reviews.append(aug_text)
    sentiments.append(row["sentiment"])

df_aug = pd.DataFrame({
    "review": aug_reviews,
    "sentiment": sentiments
})

# %%
df_aug.to_csv("data_imdb_raw/augmented_imdb.csv")

# %%
df_aug = pd.read_csv("/home/users/v-sumeth/AIS/UDA_pytorch/data_imdb_raw/augmented_imdb.csv")
df_aug = df_aug.drop(columns=["Unnamed: 0"])

# %%
tokenizer = RobertaTokenizer.from_pretrained("/home/users/v-sumeth/AIS/UDA_pytorch/tokenizers/roberta")
# %%
data = []

label_dict = {
    "positive":1,
    "negative":0
}

for _, row in tqdm(df_clean.iterrows()):
    inputs = tokenizer(row['review'],
        padding='max_length',
        truncation=True,
        max_length=256,
        return_token_type_ids=True
    )
    # inputs['input_ids'] = torch.squeeze(inputs['input_ids'])
    # inputs['attention_mask'] = torch.squeeze(inputs['attention_mask'])
    # inputs['token_type_ids'] = torch.squeeze(inputs['token_type_ids'])
    inputs['label'] = label_dict[row['sentiment']]

    data.append(inputs)
# %%
data_aug = []

for _, row in tqdm(df_aug.iterrows()):
    inputs = tokenizer(row['review'],
        padding='max_length',
        truncation=True,
        max_length=256,
        return_token_type_ids=True
    )
    # inputs['input_ids'] = torch.squeeze(inputs['input_ids'])
    # inputs['attention_mask'] = torch.squeeze(inputs['attention_mask'])
    # inputs['token_type_ids'] = torch.squeeze(inputs['token_type_ids'])
    inputs['label'] = label_dict[row['sentiment']]

    data_aug.append(inputs)

# %%
# write to text files:
# 0 - num_sup = sup train
# last 10k = sup test
# the rest is unsup

# sup train

num_sup = 5000

with open("imdb_sup_train_new.txt", "w") as f:
    f.write("input_ids\tinput_mask\tinput_type_ids\tlabel_ids\n")

    for row in range(num_sup):
        d = data[row]
        line = "["

        for x in d["input_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d["attention_mask"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d["token_type_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t"

        line += str(d["label"]) + "\n"

        f.write(line)
# %%
# unsup

with open("imdb_unsup_train_new.txt", "w") as f:
    f.write("ori_input_ids\tori_input_mask\tori_input_type_ids\taug_input_ids\taug_input_mask\taug_input_type_ids\n")

    for row in range(num_sup, len(data)-10000):
        d = data[row]
        d_aug = data_aug[row]
        line = "["

        # original
        for x in d["input_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d["attention_mask"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d["token_type_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        # augmented
        for x in d_aug["input_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d_aug["attention_mask"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d_aug["token_type_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\n"

        f.write(line)
# %%
# sup test

with open("imdb_sup_test_new.txt", "w") as f:
    f.write("input_ids\tinput_mask\tinput_type_ids\tlabel_ids\n")

    for row in range(len(data)-10000, len(data)):
        d = data[row]
        line = "["

        for x in d["input_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d["attention_mask"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t["

        for x in d["token_type_ids"]:
            line += str(x) + ", "
        line = line[0:-2]
        line += "]\t"

        line += str(d["label"]) + "\n"

        f.write(line)
# %%
