from gpt_db_tools.dbutil import DBUtil
from transformers import AutoTokenizer, AutoModel
import torch
import time
import argparse


# 开始处理问题列表，并生成内容回写到表
def start_process_question():
    print("服务已启动...")
    # 1.处理上次未回答完成或异常终止的问题
    excep_list = dbutil.query_question_list("002", args.bot_type, args.sub_type)
    for excep in excep_list:
        session, qid, bot_type, question_str, q_seq = excep[0], excep[1], excep[2], excep[3], excep[4]
        dbutil.update_question_status(qid, session, "001", q_seq)

    while True:
        question_list = dbutil.query_question_list("001", args.bot_type, args.sub_type)
        for question in question_list:
            session, qid, bot_type, question_str, q_seq = question[0], question[1], question[2], question[3], question[
                4]
            print(
                "开始处理==========>>> Session:" + session + ",问题ID:" + qid + ",机器人类型:" + bot_type + ",问题:" + question_str + ",q_seq:" + str(
                    q_seq))
            dbutil.update_question_status(qid, session, "002", q_seq)
            answer = get_response_by_bot(question_str, session, args.bot_type, args.sub_type)
            dbutil.update_question_answer_count(qid, session, q_seq, 3)
            # 更新数据表
            dbutil.update_question_answer(qid, session, "003", answer, "", "", q_seq)
        time.sleep(5)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--model_path', default="../chatglm-6b", type=str, required=False, help='模型文件路径')
    parser.add_argument('--bot_type', default="chat", type=str, required=False, help='机器人类型')
    parser.add_argument('--sub_type', default="normal", type=str, required=False, help='机器人子类型')
    return parser.parse_args()


def get_response_by_bot(input_txt, session, bot_type, sub_type):
    # 这里需要解决，每个用户不能连续对话的问题,从数据库里获取这个用户前三个问题和回答是什么
    history = []
    historyQATuple = dbutil.query_question_list_by_session(session, bot_type, sub_type)
    calc_len = len(input_txt)
    for tmp in historyQATuple:
        qaTuple = (tmp[0], tmp[1])
        calc_len = calc_len + len(tmp[0]) + len(tmp[1])
        history.append(qaTuple)
    response, history = model.chat(tokenizer, input_txt, history=history, max_length=calc_len * 5)
    torch.cuda.empty_cache()
    return response


if __name__ == '__main__':
    global args
    global dbutil
    global tokenizer
    global model
    args = set_args()
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()
    dbutil = DBUtil('192.168.10.30', '12277', 'userapp', '1Qaz2Wsx', 'ai')
    dbutil.print_db_info()
    start_process_question()
