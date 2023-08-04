import argparse
from datetime import date
import os
import time
import uuid

import pandas as pd
from transformers import T5Tokenizer

from watermark.watermark_processor import WatermarkDetector
from util import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", default="", type=str)
    parser.add_argument("--model_type", default="", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_dir", default="./log/", type=str)
    args = parser.parse_args()

    # initialize log_dir
    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    time_str = time.strftime("%H-%M-%S", time.localtime())
    datetime_str = date_str + "-" + time_str
    log_dir = os.path.join(args.log_dir, datetime_str + "-" + str(uuid.uuid4()),)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # deterministic behavior for reproducibility
    set_seed(args.seed)

    # T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_type, cache_dir=args.model_cache_dir
    )

    # watermark detector
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,  # should match original setting
        seeding_scheme="simple_1",  # should match original setting
        device="cuda",  # must match the original rng device type
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_bigrams=False,
    )

    summaries = list(pd.read_csv(args.summary_path)["summary"])
    for summary in summaries:
        score_dict = watermark_detector.detect(summary)
        print(score_dict)
        assert False


if __name__ == "__main__":
    main()
