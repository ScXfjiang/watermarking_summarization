## Apply watermarking algorithm to LLM-based text summarization model

This repository demonstrates the application of [the watermarking algorithm](https://github.com/jwkirchenbauer/lm-watermarking) to T5-based text summarization models. We present a complete guide to fine-tuning and testing the T5 summarization model using two different datasets: News Summarization Dataset and CNN-DailyMails News Dataset.

### **Steps to reproduce the work**
1. Download the dataset and divide the dataset into train set and test set if necessary.
   * [News Summary Dataset](https://www.kaggle.com/datasets/sunnysai12345/news-summary)
   * [CNN-DailyMail Newspaper Text Summarization Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
2. Fine-tune T5 summarization model:
```bash
python train.py \
    --model_type=${T5_model_type} \
    --dataset_type=${dataset_type} \
    --dataset_path="/path/to/train_set" \
    --batch_size=16 \
    --num_epoch=2 \
    --lr=1e-4 \
    --doc_max_len=512 \
    --summary_max_len=150 \
    --log_dir=${log_dir}
```
3. Test T5 summarization model:
```bash
python test.py \
    --model_type=${T5_model_type} \
    --dataset_type=${dataset_type} \
    --dataset_path="/path/to/test_set" \
    --state_dict_path="/path/to/checkpoint" \
    --batch_size=16 \
    --doc_max_len=512 \
    --summary_max_len=150 \
    --log_dir="." \
    --watermark=${enable_watermark} \
    --log_dir=${log_dir}
```

### **Experiment Results**
#### News Summary Dataset
|                               | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------------------------|:-------:|:-------:|:-------:|
| T5-base without watermarking  |  0.4832 |  0.2642 |  0.3631 |
| T5-base with watermarking     |  0.4616 |  0.2321 |  0.3345 |
| T5-large without watermarking |  **0.4901** |  **0.2697** |  **0.3632** |
| T5-large with watermarking    |  0.4780 |  0.2401 |  0.3413 |

#### CNN-DailyMail Newspaper Text Summarization Dataset
|                               | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------------------------|:-------:|:-------:|:-------:|
| T5-base without watermarking  |  **0.4174** |  **0.1957** |  **0.2961** |
| T5-base with watermarking     |  0.4031 |  0.1758 |  0.2781 |
| T5-large without watermarking |  still training |  still training |  still training |
| T5-large with watermarking    |  still training |  still training |  still training |
