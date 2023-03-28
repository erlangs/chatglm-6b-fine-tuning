#!/bin/bash
cd /home/thudm/
ps -ef|grep "chat_server.py"|grep -v "grep"|awk '{print $2}'|xargs kill -9
nohup python3 chat_server.py --model_path /home/thudm/chatglm-6b --bot_type chat --sub_type normal >> ./logs/chat_server.log 2>&1 &