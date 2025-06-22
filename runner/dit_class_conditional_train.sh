CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--config_file config/parallel/default_config \
--main_process_port 30002 \
--num_processes 8 \
runner/dit_class_conditional_train.py \
--config config/dit_on_siglip_dim_down.yaml