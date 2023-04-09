export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

#CUDA_VISIBLE_DEVICES=0 python src/composenW_diffuser.py \
#           --paths "logs/Louis_Wain/delta.bin+logs/cat/delta.bin" \
#           --categories  "Louis Wain art+cat" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --ckpt $MODEL_NAME

#CUDA_VISIBLE_DEVICES=0 python src/composenW_diffuser.py \
#           --paths "logs/Louis_Wain/delta.bin+logs/bean_curd_cat/delta.bin" \
#           --categories  "Louis Wain art+bean_curd_cat" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --ckpt $MODEL_NAME

#CUDA_VISIBLE_DEVICES=0 python src/composenW_negative_diffuser.py \
#           --paths "logs/Louis_Wain/delta.bin+logs/cat/delta.bin" \
#           --save_path "optimized_negative_captions_logs" \
#           --categories  "art+cat" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --negative_prompt "data/negative_captions.txt" \
#           --ckpt $MODEL_NAME

#CUDA_VISIBLE_DEVICES=0 python src/composenW_diffuser.py \
#           --paths "logs/cat/delta.bin+logs/wooden_pot/delta.bin" \
#           --save_path "optimized_captions_logs" \
#           --categories  "cat+wooden_pot" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --ckpt $MODEL_NAME

#CUDA_VISIBLE_DEVICES=0 python src/composenW_negative_diffuser.py \
#           --paths "logs/cat/delta.bin+logs/wooden_pot/delta.bin" \
#           --save_path "optimized_negative_captions_logs" \
#           --categories  "cat+wooden_pot" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --negative_prompt "data/negative_captions.txt" \
#           --ckpt $MODEL_NAME

CUDA_VISIBLE_DEVICES=0 python src/composenW_diffuser.py \
           --paths "logs/cat_pretrained/delta.bin+logs/wooden_pot_pretrained/delta.bin" \
           --save_path "optimized_captions_pretrained_logs" \
           --categories  "cat+wooden_pot" \
           --regularization_prompt "data/regularization_captions.txt" \
           --ckpt $MODEL_NAME

CUDA_VISIBLE_DEVICES=0 python src/composenW_negative_diffuser.py \
           --paths "logs/cat_pretrained/delta.bin+logs/wooden_pot_pretrained/delta.bin" \
           --save_path "optimized_negative_captions_pretrained_logs" \
           --categories  "cat+wooden_pot" \
           --regularization_prompt "data/regularization_captions.txt" \
           --negative_prompt "data/negative_captions.txt" \
           --ckpt $MODEL_NAME

#CUDA_VISIBLE_DEVICES=0 python src/composenW_negative_diffuser.py \
#           --paths "logs/Louis_Wain/delta.bin+logs/bean_curd_cat/delta.bin" \
#           --save_path "optimized_negative_captions_logs" \
#           --categories  "Louis Wain art+cat" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --negative_prompt "data/negative_captions_for_cat.txt" \
#           --ckpt $MODEL_NAME

#CUDA_VISIBLE_DEVICES=0 python src/composenW_diffuser.py \
#           --paths "logs/red_person/delta.bin+logs/cat/delta.bin" \
#           --save_path "optimized_logs" \
#           --categories  "red_person+cat" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --ckpt $MODEL_NAME

#CUDA_VISIBLE_DEVICES=0 python src/composenW_negative_diffuser.py \
#           --paths "logs/red_person/delta.bin+logs/cat/delta.bin" \
#           --save_path "optimized_negative_captions_logs" \
#           --categories  "red_person+cat" \
#           --regularization_prompt "data/regularization_captions.txt" \
#           --negative_prompt "data/negative_captions.txt" \
#           --ckpt $MODEL_NAME