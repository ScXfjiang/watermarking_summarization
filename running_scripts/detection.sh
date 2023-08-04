python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/cnn_T5_base_watermark_false/Aug-01-2023-21-27-47-04a52ead-4a5d-444f-9ff5-1d6c56e93a51/summary_target.csv" \
    --model_type="t5-base" \
    --log_dir="cnn_T5_base_watermark_false"

python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/cnn_T5_base_watermark_true/Aug-01-2023-21-28-01-4066f99d-5415-4212-bec0-4e441968eaad/summary_target.csv" \
    --model_type="t5-base" \
    --log_dir="cnn_T5_base_watermark_true"

python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/cnn_T5_large_watermark_false/Aug-02-2023-11-11-50-014154f9-7215-443f-9adf-d584d199e1f0/summary_target.csv" \
    --model_type="t5-large" \
    --log_dir="cnn_T5_large_watermark_false"

python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/cnn_T5_large_watermark_true/Aug-02-2023-12-38-08-c5ef44bc-22a9-4b49-a7be-ce98bf579c66/summary_target.csv" \
    --model_type="t5-large" \
    --log_dir="cnn_T5_large_watermark_true"

python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/news_T5_base_watermark_False/Aug-02-2023-11-52-25-f9b1424f-3db0-4404-a7df-4880ae8771f2/summary_target.csv" \
    --model_type="t5-base" \
    --log_dir="news_T5_base_watermark_False"

python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/news_T5_base_watermark_true/Aug-02-2023-12-03-56-40125473-035a-49ce-870a-ba176b3594ac/summary_target.csv" \
    --model_type="t5-base" \
    --log_dir="news_T5_base_watermark_true"

python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/news_T5_large_watermark_False/Aug-02-2023-12-15-52-9a6203e7-25a7-474d-9688-111481986760/summary_target.csv" \
    --model_type="t5-large" \
    --log_dir="news_T5_large_watermark_False"

python ../detection.py \
    --summary_path="/scratch/22200056/T5_pretrained_model/inference_log/news_T5_large_watermark_true/Aug-02-2023-12-30-52-b3b79d73-5c7d-4629-a8e1-5bb9fae74a72/summary_target.csv" \
    --model_type="t5-large" \
    --log_dir="news_T5_large_watermark_true"