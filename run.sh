CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--config_file config/parallel/default_config \
--main_process_port 30002 \
--num_processes 2 \
run.py