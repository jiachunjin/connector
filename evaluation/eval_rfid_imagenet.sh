accelerate launch --config_file \
config/parallel/default_config --main_process_port 30002 \
evaluation/eval_rfid_imagenet.py