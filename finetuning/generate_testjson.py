import json
import os

"""
    生成测试json文件
    author:chen.yiwan
"""
json_file_path = "../data/idigital.json"
train_json = []
for i in range(1500):
    train_json.append({'instruction': 'who are you,who you are', 'input': '你是谁',
                       'output': '我是一名AI人工智能助手，是由[艾瑞数智]于 2023 年训练的语言模型。我的任务是针对用户的问题和要求提供适当的答复和支持。'})
    train_json.append({'instruction': 'who are you,who you are', 'input': 'who are you?',
                       'output': 'I am an AI smart assistant, a language model trained by iDigital in 2023. My mission is to provide appropriate responses and support to users questions and requests.'})

json_data = json.dumps(train_json, separators=(',', ': '))
if os.path.exists(json_file_path):
    os.remove(json_file_path)
f = open(json_file_path, 'w')
f.write(json_data)
f.close()
