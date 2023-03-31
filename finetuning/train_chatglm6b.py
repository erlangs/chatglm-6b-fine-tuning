import os
from typing import Optional

from MyTrainer import Trainer
from transformers import TrainingArguments
import random
import argparse
from glob import glob
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import torch


def get_masks_and_position_ids(
        seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
            seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position
    return attention_mask, position_ids


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
    parser.add_argument('--dataset_path', default='./data/*', type=str, required=False, help='数据集目录')
    parser.add_argument('--model_path', default="../chatglm-6b", type=str, required=False,
                        help='原始发布的预训练模型目录')
    parser.add_argument('--save_model_path', default="../save_model_path", type=str, required=False,
                        help='微调模型保存目录')
    return parser.parse_args()


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [-100] * (seq_len - 1)
                + ids[(seq_len - 1):]
                + [tokenizer.eos_token_id]
                + [-100] * (longest - ids_l - 1)
        )
        ids = ids + [tokenizer.eos_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def preprocess(example):
    max_seq_length = 512
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    # {"context": context, "target": target}
    example['context'] = context
    example['target'] = target
    return example


def filter_nan(example):
    return example['target'] is not None and example['context'] is not None


class MyTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        def save_tunable_parameters(model, path):
            saved_params = {
                k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
            }
            # saved_params = model.state_dict()
            torch.save(saved_params, path)

        save_tunable_parameters(
            self.model, os.path.join(output_dir, "chatglm-lora.pt")
        )


def start_train(run_args):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(run_args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(run_args.model_path, trust_remote_code=True).half().cuda()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=['query_key_value', ],
    )
    model = get_peft_model(model, peft_config)

    random.seed(42)
    all_file_list = glob(pathname=run_args.dataset_path)
    test_file_list = random.sample(all_file_list,
                                   int(1 if (len(all_file_list) * 0.25 < 1) else len(all_file_list) * 0.25))
    train_file_list = [i for i in all_file_list if i not in test_file_list]
    if len(train_file_list) <= 0:
        train_file_list = test_file_list

    dataset = load_dataset(
        "csv",
        data_files={
            'train': train_file_list,
            'valid': test_file_list
        },
        cache_dir="cache_data"
    )
    tokenized_datasets = dataset.map(function=format_example, remove_columns=dataset['train'].column_names).filter(
        function=filter_nan)
    tokenized_datasets = tokenized_datasets.map(function=preprocess)

    args = TrainingArguments(
        output_dir=run_args.save_model_path,
        per_device_train_batch_size=1,  # 如果在24G显存以上的显卡，可以开到4
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=50,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False
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
