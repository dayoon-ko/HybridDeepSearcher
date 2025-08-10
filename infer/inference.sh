#! /bin/bash

for dataset in musique fanoutqa frames med_browse_comp; 
do
    python inference.py \
        --dataset_name $dataset \
        --model_id_or_path dayoon/Qwen3-8B-sft \
        --top_k 5
done

for dataset in browse_comp;
do
    python inference.py \
        --dataset_name $dataset \
        --model_id_or_path dayoon/Qwen3-8B-sft \
        --use_jina True \
        --max_doc_len 1024 \
        --top_k 10
done