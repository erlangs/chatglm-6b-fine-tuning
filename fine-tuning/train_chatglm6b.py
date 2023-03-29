from MyTrainer import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
import random
import argparse
from glob import glob
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType


def tokenize(element):
    context_length = 512
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data2/*', type=str, required=False, help='数据集目录')
    parser.add_argument('--model_path', default="../chatglm-6b", type=str, required=False,
                        help='原始发布的预训练模型目录')
    parser.add_argument('--save_model_path', default="../save_model_path", type=str, required=False,
                        help='微调模型保存目录')
    return parser.parse_args()


def start_train(run_args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(run_args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(run_args.model_path, trust_remote_code=True).half().cuda()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=['query_key_value', ],
    )
    model = get_peft_model(model, peft_config)
    random.seed(42)
    all_file_list = glob(pathname=run_args.dataset_path)
    test_file_list = random.sample(all_file_list, int(len(all_file_list) * 0.05))
    train_file_list = [i for i in all_file_list if i not in test_file_list]
    raw_datasets = load_dataset("csv", data_files={'train': train_file_list, 'valid': test_file_list},
                                cache_dir="cache_data")
    tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=run_args.save_model_path,
        per_device_train_batch_size=3,  # 如果在24G显存上的显卡，可以开到4
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=100,
        fp16=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    print("start train...")
    trainer.train()


if __name__ == '__main__':
    global run_args
    run_args = set_args()
    start_train(run_args)
