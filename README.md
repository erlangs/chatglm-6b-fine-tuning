### 查看此文档，你将学到什么?
+ 1 如何部署运行chatglm-6b
+ 2 如何微调模型

#### 如何部署运行chatglm-6b?

- 1、git clone https://huggingface.co/THUDM/chatglm-6b.git
- 2、安装依赖
  ~~~
  pip3 install protobuf==3.20.0 transformers==4.26.1 icetk cpm_kernels
  ~~~
- 3、修改 chat_interact.py 中的 PRE_TRAINED_MODEL_PATH='上面克隆的chatglm-6b文件夹路径'
- 4、运行代码 python3 chat_interact.py
- 5、生成效果如下：
  回答问题
    - ![1.png](images%2F1.png)
      做数学题
    - ![2.png](images%2F2.png)
      写标书提纲
    - ![3.png](images%2F3.png)
      时政解答
    - ![4.png](images%2F4.png)
      翻译
    - ![5.png](images%2F5.png)
    - 另外还有许多功能，比如：
        - 自我认知
        - 提纲写作
        - 文案写作
        - 邮件写作助手
        - 信息抽取
        - 角色扮演
        - 评论比较
        - 旅游向导

#### 如何微调chatglm-6b?
+ 1 准备数据集
+ 2 运行train_chatglm6b.py 训练代码

#### 文件结构说明

+ chat_interact.py 交互式对话，命令行下运行,一般用于测试机器人对话。
+ chat_server.py 连接数据库，根据数据表对话内容，排队进行回答，并将生成内容回写到表，它依赖我写的数据库连接组件，另外还有一个http服务接收前端请求。
+ start_chat_server.sh 启动chat_server.py
+ data2 训练数据集
+ train_chatglm6b.py 训练代码
  ~~~
  训练需要安装如下依赖
  pip3 install datasets
  pip3 install peft
  ~~~
  
#### 环境说明

+ 系统版本：CentOS Linux release 7.9.2009 (Core)
+ 内核版本：3.10.0-1160.el7.x86_64
+ python 版本：3.7.16
+ NVIDIA驱动版本： 515.65.01
+ CUDA 版本：11.7
+ cuDNN 版本：v8.8.0
+ GPU：3090 24GB