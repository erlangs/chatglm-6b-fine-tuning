import json
import os

"""
    生成测试json文件
    author:chen.yiwan
    date:2023-03-31
"""
json_file_path = "../data/idigital_1.json"
train_json = []
for i in range(1500):
    train_json.append({'instruction': 'who are you,who you are', 'input': '你是谁',
                       'output': '我是一名AI人工智能助手，是由[艾瑞数智]于 2023 年训练的语言模型。我的任务是针对用户的问题和要求提供适当的答复和支持。'})
    train_json.append({'instruction': 'who are you,who you are', 'input': 'who are you?',
                       'output': 'I am an AI smart assistant, a language model trained by iDigital in 2023. My mission is to provide appropriate responses and support to users questions and requests.'})
    train_json.append({'instruction': '李强总理的简历', 'input': '李强总理的简历',
                       'output': '李强，男，汉族，1959年7月生，浙江瑞安人，1976年7月参加工作，1983年4月加入中国共产党，中央党校研究生学历，高级管理人员工商管理硕士学位。现任中共二十届中央政治局常委，国务院总理、党组书记。'})

json_data = json.dumps(train_json, separators=(',', ': '))
if os.path.exists(json_file_path):
    os.remove(json_file_path)
f = open(json_file_path, 'w')
f.write(json_data)
f.close()
