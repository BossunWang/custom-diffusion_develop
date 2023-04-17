export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

#python src/convert.py \
#      --delta_ckpt logs/cat_pretrained/delta_cat.ckpt \
#      --mode compvis-to-diffuser \
#      --ckpt ../stable-diffusion/sd-v1-4.ckpt
#
#python src/convert.py \
#      --delta_ckpt logs/wooden_pot_pretrained/delta_wooden_pot.ckpt \
#      --mode compvis-to-diffuser \
#      --ckpt ../stable-diffusion/sd-v1-4.ckpt

python src/convert.py \
      --delta_ckpt logs/dog_pretrained/delta_dog.ckpt \
      --mode compvis-to-diffuser \
      --ckpt ../stable-diffusion/sd-v1-4.ckpt
