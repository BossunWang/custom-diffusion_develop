export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

# launch training script (2 GPUs recommended, increase --max_train_steps to 500 if 1 GPU)


CUDA_VISIBLE_DEVICES=0  accelerate launch src/diffuser_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs/Louis_Wain_bean_curd_cat  \
          --concepts_list=./assets/concept_list_Louis_Wain_bean_curd_cat.json \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --resolution=256  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>+<new2>" \
          --mixed_precision="fp16" \
          --use_8bit_adam \
          --gradient_accumulation_steps=1

