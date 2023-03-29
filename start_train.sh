#!/bin/bash
cd /home/thudm/chatglm-6b-fine-tuning
nohup python3 ./fine-tuning/train_chatglm6b.py --model_path /home/thudm/chatglm-6b-fine-tuning/fine-tuning/pre-trained-model/ --dataset_path "/home/thudm/data/*" >>./logs/train.log 2>&1 &
