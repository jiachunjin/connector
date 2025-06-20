#!/bin/bash

echo "开始测试 RobustDataLoader..."

CUDA_VISIBLE_DEVICES=0 accelerate launch \
--config_file config/parallel/default_config \
--main_process_port 30003 \
--num_processes 1 \
runner/test_robust_dataloader.py

echo "RobustDataLoader 测试完成" 