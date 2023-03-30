from transformers import AutoTokenizer
from model.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType

model = ChatGLMForConditionalGeneration.from_pretrained("./model")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value', ],
)
model = get_peft_model(model, peft_config)

peft_path = "G:\\save_model\\chatglm-lora.pt"
model.load_state_dict(torch.load(peft_path), strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
text = "？"

with torch.autocast("cuda"):
    res, history = model.chat(tokenizer=tokenizer, query=text, max_length=300)
    print(res)
