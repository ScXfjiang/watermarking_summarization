export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --model_type="t5-base" \
    --dataset_type="cnn" \
    --dataset_path="/scratch/22200056/dataset/cnn_dailymail/test.csv" \
    --state_dict_path="/scratch/22200056/T5_pretrained_model/T5_base/cnn.pt" \
    --batch_size=64 \
    --doc_max_len=512 \
    --summary_max_len=150 \
    --watermark="True" \
    --log_dir="cnn_T5_base_watermark_true" \
