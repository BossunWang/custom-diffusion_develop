export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

# launch training script (2 GPUs recommended, increase --max_train_steps to 500 if 1 GPU)

CUDA_VISIBLE_DEVICES=0 accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/cat  \
          --class_data_dir=./real_reg/samples_tabby_cat/ \
          --output_dir=./logs/cat  \
          --with_prior_preservation \
          --real_prior --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> cat"  \
          --class_prompt="cat" \
          --resolution=256  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr \
          --hflip  \
          --modifier_token "<new1>" \
          --mixed_precision="fp16" \
          --revision fp16 \
          --use_8bit_adam \
          --gradient_accumulation_steps=1

CUDA_VISIBLE_DEVICES=0 accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/wooden_pot  \
          --class_data_dir=./real_reg/samples_wooden_pot/ \
          --output_dir=./logs/wooden_pot  \
          --with_prior_preservation \
          --real_prior --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> wooden pot"  \
          --class_prompt="wooden pot" \
          --resolution=256  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr \
          --hflip  \
          --modifier_token "<new1>" \
          --mixed_precision="fp16" \
          --revision fp16 \
          --use_8bit_adam \
          --gradient_accumulation_steps=1

#CUDA_VISIBLE_DEVICES=0 accelerate launch src/diffusers_training.py \
#          --pretrained_model_name_or_path=$MODEL_NAME  \
#          --instance_data_dir=./data/Louis_Wain_cat  \
#          --class_data_dir=./real_reg/samples_oil_painting/ \
#          --output_dir=./logs/Louis_Wain  \
#          --with_prior_preservation \
#          --real_prior --prior_loss_weight=1.0 \
#          --instance_prompt="painting of <new1> art"  \
#          --class_prompt="oil painting" \
#          --resolution=256  \
#          --train_batch_size=2  \
#          --learning_rate=1e-5  \
#          --lr_warmup_steps=0 \
#          --max_train_steps=500 \
#          --num_class_images=200 \
#          --scale_lr \
#          --hflip  \
#          --modifier_token "<new1>" \
#          --mixed_precision="fp16" \
#          --revision fp16 \
#          --use_8bit_adam \
#          --gradient_accumulation_steps=1

#CUDA_VISIBLE_DEVICES=0 accelerate launch src/diffusers_training.py \
#          --pretrained_model_name_or_path=$MODEL_NAME  \
#          --instance_data_dir=./data/bean_curd_cat  \
#          --class_data_dir=./real_reg/samples_tabby_cat/ \
#          --output_dir=./logs/bean_curd_cat  \
#          --with_prior_preservation \
#          --real_prior --prior_loss_weight=1.0 \
#          --instance_prompt="photo of a <new1> cat"  \
#          --class_prompt="tabby cat" \
#          --resolution=256  \
#          --train_batch_size=2  \
#          --learning_rate=1e-5  \
#          --lr_warmup_steps=0 \
#          --max_train_steps=500 \
#          --num_class_images=200 \
#          --scale_lr \
#          --hflip  \
#          --modifier_token "<new1>" \
#          --mixed_precision="fp16" \
#          --revision fp16 \
#          --use_8bit_adam \
#          --gradient_accumulation_steps=1