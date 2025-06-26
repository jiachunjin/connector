CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--config_file config/parallel/default_config \
--main_process_port 30002 \
--num_processes 4 \
runner/vit_decoder_train.py \
--config config/vit_decoder_scale.yaml