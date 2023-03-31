#!/bin/bash
cd /home/thudm/chatglm-6b-fine-tuning
nohup python3 ./fine-tuning/fine_tuning_chatglm6b.py --model_path /home/thudm/chatglm-6b-fine-tuning/fine-tuning/pre-trained-model/ --dataset_path "/home/thudm/data/*" --batch_size 4 --fp16 >>./logs/train.log 2>&1 &
