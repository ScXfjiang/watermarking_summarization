export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --model_type="t5-base" \
    --dataset_type="news" \
    --dataset_path="/scratch/22200056/dataset/news_summary/test.csv" \
    --state_dict_path="/scratch/22200056/T5_pretrained_model/T5_base/news_sum.pt" \
    --batch_size=16 \
    --doc_max_len=512 \
    --summary_max_len=150 \
    --log_dir="." \
    --watermark="False"
