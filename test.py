import argparse
from datetime import date
import os
import time
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, LogitsProcessorList
from rouge_score import rouge_scorer

from watermark.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from dataset import news_sum_dataset
from dataset import cnn_daily_mail_dataset
from util import set_seed


def evaluate(model, data_loader, summary_max_len, tokenizer, watermark):
    model.eval()
    summaries = []
    targets = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # watermark processor
    if watermark:
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.25,
            delta=2.0,
            seeding_scheme="simple_1",
        )

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            target_ids = data["target_ids"].to("cuda", dtype=torch.long)
            doc_ids = data["doc_ids"].to("cuda", dtype=torch.long)
            doc_mask = data["doc_mask"].to("cuda", dtype=torch.long)

            generated_ids = model.generate(
                input_ids=doc_ids,
                attention_mask=doc_mask,
                max_length=summary_max_len,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                logits_processor=LogitsProcessorList([watermark_processor])
                if watermark
                else None,
            )
            summary = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in target_ids
            ]
            if batch_idx % 10 == 0:
                print("Val Completed: {}".format(batch_idx))

            summaries.extend(summary)
            targets.extend(target)

            # Compute and store ROUGE scores
            for pred, act in zip(summary, target):
                scores = scorer.score(pred, act)
                rouge1_scores.append(scores["rouge1"].fmeasure)
                rouge2_scores.append(scores["rouge2"].fmeasure)
                rougeL_scores.append(scores["rougeL"].fmeasure)

    return (
        summaries,
        targets,
        np.mean(rouge1_scores),
        np.mean(rouge2_scores),
        np.mean(rougeL_scores),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="", type=str)
    parser.add_argument("--dataset_type", default="", type=str)
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--state_dict_path", default="", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--doc_max_len", default=512, type=int)
    parser.add_argument("--summary_max_len", default=150, type=int)
    parser.add_argument("--watermark", default="", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--log_dir", default="/scratch/22200056/watermark_log", type=str
    )
    parser.add_argument(
        "--model_cache_dir", default="/home/people/22200056/scratch/cache", type=str
    )
    args = parser.parse_args()

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
    tokenizer = T5Tokenizer.from_pretrained(args.model_type, cache_dir=args.model_cache_dir)

    # dataset and dataloader
    if args.dataset_type == "news":
        dataset_fn = news_sum_dataset
    elif args.dataset_type == "cnn":
        dataset_fn = cnn_daily_mail_dataset
    else:
        raise ValueError("Invalid dataset type: {}".format(args.dataset_type))
    test_set = dataset_fn(
        args.dataset_path, tokenizer, args.doc_max_len, args.summary_max_len
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # T5 model and load fine tuned weights
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_type, cache_dir=args.model_cache_dir
    )
    state_dict = torch.load(args.state_dict_path)
    model.load_state_dict(state_dict)
    model = model.to("cuda")

    # evaluate
    summaries, targets, avg_rouge1, avg_rouge2, avg_rougeL = evaluate(
        model,
        test_loader,
        args.summary_max_len,
        tokenizer,
        args.watermark.lower() == "true",
    )
    with open(os.path.join(log_dir, "rouge.txt"), "a") as f:
        f.write("Validation ROUGE-1: {}\n".format(avg_rouge1))
        f.write("Validation ROUGE-2: {}\n".format(avg_rouge2))
        f.write("Validation ROUGE-L: {}\n".format(avg_rougeL))
    summary_df = pd.DataFrame({"summary": summaries, "target": targets})
    summary_df.to_csv(os.path.join(log_dir, "summary_target.csv"))


if __name__ == "__main__":
    main()
