import argparse
from datetime import date
import os
import time
import uuid

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import wandb

from dataset import news_sum_dataset
from dataset import cnn_daily_mail_dataset
from util import set_seed


def train_epoch(model, data_loader, optimizer, tokenizer):
    model.train()
    for idx, data in enumerate(data_loader, 0):
        # [batch_size, max_news_len]
        doc_ids = data["doc_ids"].to("cuda", dtype=torch.long)
        # [batch_size, max_news_len]
        doc_mask = data["doc_mask"].to("cuda", dtype=torch.long)
        # target summery: [batch_size, max_sum_len]
        target_ids = data["target_ids"].to("cuda", dtype=torch.long)
        # teacher forcing strategy
        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_labels = target_ids[:, 1:].clone().detach()
        decoder_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100
        # forward
        loss = model(
            input_ids=doc_ids,
            attention_mask=doc_mask,
            decoder_input_ids=decoder_input_ids,
            labels=decoder_labels,
        )[0]
        # logging
        if idx % 10 == 0:
            wandb.log({"Training Loss": loss.item()})
        # backward
        optimizer.zero_grad()
        loss.backward()
        # model update
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="", type=str)
    parser.add_argument("--dataset_type", default="", type=str)
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_epoch", default=2, type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--doc_max_len", default=512, type=int)
    parser.add_argument("--summary_max_len", default=150, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--model_cache_dir", default="/home/people/22200056/scratch/cache", type=str
    )
    parser.add_argument(
        "--log_dir", default="/scratch/22200056/watermark_log", type=str
    )
    args = parser.parse_args()

    # logging with wandb
    wandb.init(project="watermarking doc summarization")
    config = wandb.config
    config.model_type = args.model_type
    config.dataset_type = args.dataset_type
    config.batch_size = args.batch_size
    config.num_epoch = args.num_epoch
    config.lr = args.lr
    config.doc_max_len = args.doc_max_len
    config.summary_max_len = args.summary_max_len
    config.seed = args.seed

    # initialize log_dir
    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    time_str = time.strftime("%H-%M-%S", time.localtime())
    datetime_str = date_str + "-" + time_str
    log_dir = os.path.join("log", args.log_dir, datetime_str + "-" + str(uuid.uuid4()),)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # deterministic behavior for reproducibility
    set_seed(args.seed)

    # T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_type, cache_dir=args.model_cache_dir
    )

    # dataset and dataloader
    if args.dataset_type == "news":
        dataset_fn = news_sum_dataset
    elif args.dataset_type == "cnn":
        dataset_fn = cnn_daily_mail_dataset
    else:
        raise ValueError("Invalid dataset type: {}".format(args.dataset_type))
    train_set = dataset_fn(
        args.dataset_path, tokenizer, args.doc_max_len, args.summary_max_len
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # T5 model
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_type, cache_dir=args.model_cache_dir
    )
    model = model.to("cuda")
    # log all metrics
    wandb.watch(model, log="all")

    # Adam optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # train
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for epoch in range(args.num_epoch):
        print("Training Epoch: {}".format(epoch))
        train_epoch(model, train_loader, optimizer, tokenizer)
        # save checkpoint after each epoch
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(epoch)),
        )
    # save final checkpoing
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pt"))


if __name__ == "__main__":
    main()
