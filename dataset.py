import torch
from torch.utils.data import Dataset

import pandas as pd


class CNNDailyMail(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len
        self.article = self.data.article
        self.highlights = self.data.highlights

    def __len__(self):
        return len(self.article)

    def __getitem__(self, index):
        article = str(self.article[index])
        article = " ".join(article.split())

        highlight = str(self.highlights[index])
        highlight = " ".join(highlight.split())

        source = self.tokenizer.batch_encode_plus(
            [article],
            max_length=self.source_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [highlight],
            max_length=self.summ_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        doc_ids = source["input_ids"].squeeze()
        doc_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "doc_ids": doc_ids.to(dtype=torch.long),
            "doc_mask": doc_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
        }


def cnn_daily_mail_dataset(dataset_path, tokenizer, doc_max_len, summary_max_len):
    df = pd.read_csv(dataset_path, encoding="latin-1",)
    df = df[["article", "highlights"]]
    df.highlights = "summarize: " + df.highlights
    dataset = CNNDailyMail(df, tokenizer, doc_max_len, summary_max_len)
    return dataset


class NewsSum(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = " ".join(ctext.split())

        text = str(self.text[index])
        text = " ".join(text.split())

        source = self.tokenizer.batch_encode_plus(
            [ctext],
            max_length=self.source_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        doc_ids = source["input_ids"].squeeze()
        doc_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "doc_ids": doc_ids.to(dtype=torch.long),
            "doc_mask": doc_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
        }


def news_sum_dataset(dataset_path, tokenizer, doc_max_len, summary_max_len):
    df = pd.read_csv(dataset_path, encoding="latin-1",)
    df = df[["text", "ctext"]]
    df.ctext = "summarize: " + df.ctext
    dataset = NewsSum(df, tokenizer, doc_max_len, summary_max_len)
    return dataset
