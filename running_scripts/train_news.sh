export CUDA_VISIBLE_DEVICES=0

python ../train.py \
    --model_type="t5-base" \
    --dataset_type="news" \
    --dataset_path="/scratch/22200056/dataset/news_summary/train.csv" \
    --batch_size=16 \
    --num_epoch=2 \
    --lr=1e-4 \
    --doc_max_len=512 \
    --summary_max_len=150
