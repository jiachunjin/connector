CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--config_file config/parallel/default_config \
--main_process_port 30002 \
--num_processes 8 \
runner/run.py \
--config config/learn_to_use.yaml