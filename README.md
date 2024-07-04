# DIFT
Finetuning Generative Large Language Models with Discrimination Instructions for Knowledge Graph Completion, ISWC 2024

## requirements
    pytorch==2.1.0
    bitsandbytes==0.40.0
    transformers==4.31.0
    peft==0.4.0
    accelerate==0.21.0
    einops==0.6.1
    evaluate==0.4.0
    scikit-learn==1.2.2
    sentencepiece==0.1.99
    wandb==0.15.3

## Finetuning
### FB15K237
    bash script/train_fb.sh {TransE|SimKGC|CoLE}

### WN18RR
    bash script/train_wn.sh {TransE|SimKGC|CoLE}

## Inference
    bash script/eval.sh {FB15K237|WN18RR} {TransE|SimKGC|CoLE} {checkpoint_dir}

checkpoint_dir is the path of the folder to save the PEFT model, like "./output/FB15K237/2024xxxx-xxxxxx/checkpoint-xxxx/adapter_model"
