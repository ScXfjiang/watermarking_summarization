export CUDA_VISIBLE_DEVICES=1

python ../train.py \
    --model_type="t5-base" \
    --dataset_type="cnn" \
    --dataset_path="/scratch/22200056/dataset/cnn_dailymail/train.csv" \
    --batch_size=32 \
    --num_epoch=1 \
    --lr=1e-4 \
    --doc_max_len=512 \
    --summary_max_len=150
