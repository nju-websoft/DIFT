#!/usr/bin/env bash

TASK="FB15K237"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

KGE_MODEL="CoLE"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    KGE_MODEL=$1
    shift
fi

SAVED_DIR=""
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SAVED_DIR=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/dataset/${TASK}"
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
python eval.py \
    --model_class KGELlama \
    --kge_model "${KGE_MODEL}" \
    --embedding_dim 768 \
    --train_path "${DATA_DIR}/${KGE_MODEL}/data_KGELlama/train.json" \
    --eval_path "${DATA_DIR}/${KGE_MODEL}/data_KGELlama/valid.json" \
    --test_path "${DATA_DIR}/${KGE_MODEL}/data_KGELlama/test.json" \
    --checkpoint_dir "${SAVED_DIR}" \
    --full_finetune False \
    --model_name_or_path "llama-2-7b-chat-hf" \
    --dataset "${DATA_DIR}" \
    --source_max_len 2048 \
    --target_max_len 64 \
    --max_new_tokens 64 \
    --min_new_tokens 1 \
    --do_sample False \
    --num_beams 1 \
    --num_return_sequences 1 \
    --output_scores False

    
