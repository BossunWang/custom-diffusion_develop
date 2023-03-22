export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/Louis_Wain_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "Louis Wain <new1> cat playing with a ball"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/Louis_Wain_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "<new1> Louis Wain cat standing up"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/Louis_Wain_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "a <new1> Louis Wain cat playing with a pig"

#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/bean_curd_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "a <new1> Louis Wain cat playing with a pig"

#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/bean_curd_Louis_Wain_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "a <new1> Louis Wain cat playing with a pig"

#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/ocelot/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "a <new1> Louis Wain ocelot cat playing with a pig"

#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/ocelot/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "a <new1> Louis Wain ocelot cat standing up"

#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/Louis_Wain_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "Painting of a cat wearing sunglasses in the style of <new1> art"
#
#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/Louis_Wain_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "Painting of a cat playing with ball in the style of <new1> art"

#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/Louis_Wain_bean_curd_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "Painting of <new1> cat wearing sunglasses by <new2> art"

#CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
#          --delta_ckpt logs/bean_curd_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "Painting of <new1> cat at a beach by artist claude monet"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "<new1> cat playing with a ball"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "Painting of <new1> cat at a beach by artist claude monet"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "a depressed <new1> cat"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "a <new1> cat in Taipei street"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt logs/Louis_Wain/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "painting of a cat in the style of <new1> art"

CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
          --delta_ckpt optimized_logs/optimized_Louis_Wain_art+cat/delta.bin \
          --ckpt $MODEL_NAME \
          --prompt "<new1> art style painting of <new2> cat"

CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
          --delta_ckpt optimized_logs/optimized_Louis_Wain_art+cat/delta.bin \
          --ckpt $MODEL_NAME \
          --prompt "<new1> art style painting of <new2> cat sitting on toilet"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt optimized_logs/optimized_Louis_Wain_art+bean_curd_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "<new1> art style painting of <new2> cat"

#CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
#          --delta_ckpt optimized_logs/optimized_Louis_Wain_art+bean_curd_cat/delta.bin \
#          --ckpt $MODEL_NAME \
#          --prompt "<new1> art style painting of <new2> cat sitting on toilet"