export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

# launch training script (2 GPUs recommended, increase --max_train_steps to 500 if 1 GPU)

#CUDA_VISIBLE_DEVICES=0 accelerate launch src/diffuser_training.py \
#          --pretrained_model_name_or_path=$MODEL_NAME  \
#          --instance_data_dir=/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/diffusers/examples/dreambooth/train_instance_cat  \
#          --class_data_dir=./real_reg/samples_Louis_Wain_cat/ \
#          --output_dir=./logs/Louis_Wain_cat  \
#          --with_prior_preservation \
#          --real_prior --prior_loss_weight=1.0 \
#          --instance_prompt="photo of a <new1> cat"  \
#          --class_prompt="Louis Wain cat" \
#          --resolution=256  \
#          --train_batch_size=2  \
#          --learning_rate=1e-5  \
#          --lr_warmup_steps=0 \
#          --max_train_steps=500 \
#          --num_class_images=180 \
#          --scale_lr \
#          --hflip  \
#          --modifier_token "<new1>" \
#          --mixed_precision="fp16" \
#          --revision fp16 \
#          --use_8bit_adam \
#          --gradient_accumulation_steps=1

#CUDA_VISIBLE_DEVICES=1 accelerate launch src/diffuser_training.py \
#          --pretrained_model_name_or_path=$MODEL_NAME  \
#          --instance_data_dir=./bean_curd_cat  \
#          --class_data_dir=./real_reg/samples_cat/ \
#          --output_dir=./logs/bean_curd_cat  \
#          --with_prior_preservation \
#          --real_prior --prior_loss_weight=1.0 \
#          --instance_prompt="photo of a <new1> cat"  \
#          --class_prompt="cat" \
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

#CUDA_VISIBLE_DEVICES=1 accelerate launch src/diffuser_training.py \
#          --pretrained_model_name_or_path=$MODEL_NAME  \
#          --instance_data_dir=./bean_curd_cat  \
#          --class_data_dir=./real_reg/samples_cat/ \
#          --output_dir=./logs/bean_curd_Louis_Wain_cat  \
#          --with_prior_preservation \
#          --real_prior --prior_loss_weight=1.0 \
#          --instance_prompt="photo of a <new1> cat"  \
#          --class_prompt="Louis Wain cat" \
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

#CUDA_VISIBLE_DEVICES=1 accelerate launch src/diffuser_training.py \
#          --pretrained_model_name_or_path=$MODEL_NAME  \
#          --instance_data_dir=./ocelot  \
#          --class_data_dir=./real_reg/samples_ocelot/ \
#          --output_dir=./logs/ocelot  \
#          --with_prior_preservation \
#          --real_prior --prior_loss_weight=1.0 \
#          --instance_prompt="photo of a <new1> ocelot cat"  \
#          --class_prompt="ocelot" \
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

CUDA_VISIBLE_DEVICES=1 accelerate launch src/diffuser_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/Louis_Wain_cat  \
          --class_data_dir=./real_reg/samples_Louis_Wain/ \
          --output_dir=./logs/Louis_Wain_cat  \
          --with_prior_preservation \
          --real_prior --prior_loss_weight=1.0 \
          --instance_prompt="paniting of a <new1> art"  \
          --class_prompt="Louis Wain" \
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