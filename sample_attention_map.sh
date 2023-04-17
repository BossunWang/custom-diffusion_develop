export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt logs/cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> cat"
#
#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt logs/wooden_pot/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> wooden pot"
#
#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt logs/cat_pretrained/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> cat"
#
#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt logs/wooden_pot_pretrained/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> wooden pot"
#
#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt optimized_captions_logs/optimized_cat+wooden_pot/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> cat sitting inside a <new2> wooden pot and looking up"
#
#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt optimized_negative_captions_logs/optimized_cat+wooden_pot/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> cat sitting inside a <new2> wooden pot and looking up"
#
#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt optimized_captions_pretrained_logs/optimized_cat+wooden_pot/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> cat sitting inside a <new2> wooden pot and looking up"
#
#CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
#          --delta_ckpt optimized_negative_captions_pretrained_logs/optimized_cat+wooden_pot/delta.bin \
#          --ckpt $MODEL_NAME \
#          --batch_size 3 \
#          --prompt "photo of a <new1> cat sitting inside a <new2> wooden pot and looking up"


CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
          --delta_ckpt optimized_captions_pretrained_logs/optimized_cat+dog/delta.bin \
          --ckpt $MODEL_NAME \
          --batch_size 3 \
          --prompt "a <new1> cat and a <new2> dog playing together"

CUDA_VISIBLE_DEVICES=0 python src/show_attention_map_diffuser.py \
          --delta_ckpt optimized_negative_captions_pretrained_logs/optimized_cat+dog/delta.bin \
          --ckpt $MODEL_NAME \
          --batch_size 3 \
          --prompt "a <new1> cat and a <new2> dog playing together"