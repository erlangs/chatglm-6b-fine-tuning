from glob import glob
import os
import pandas as pd
import shutil
from itertools import chain
from tqdm import tqdm
from pathlib import Path

target_dir_list = ['G:\\idea_work2\\alpaca_chinese_dataset-main\\其他中文问题补充\\',
                   'G:\\idea_work2\\alpaca_chinese_dataset-main\\alpaca_chinese_dataset\\翻译后的中文数据\\',
                   'G:\\idea_work2\\alpaca_chinese_dataset-main\\alpaca_chinese_dataset\\chatglm问题数据补充\\'
                   ]

all_json_path = [glob(i + "*.json") for i in target_dir_list]
all_json_path = list(chain(*all_json_path))


def read_json(x: str):
    try:
        data = pd.read_json(x)
        return data
    except Exception as e:
        return pd.DataFrame()


alldata = pd.concat([read_json(i) for i in all_json_path])

generate_date_dir = "g:\\data"
generate_date_dir = Path(generate_date_dir)

if generate_date_dir.exists():
    shutil.rmtree(generate_date_dir, ignore_errors=True)

os.makedirs(generate_date_dir, exist_ok=True)
alldata = alldata.sample(frac=1).reset_index(drop=True)

chunk_size = 666

for index, start_id in tqdm(enumerate(range(0, alldata.shape[0], chunk_size))):
    temp_data = alldata.iloc[start_id:(start_id + chunk_size)]
    temp_data.to_csv(generate_date_dir.joinpath(f"{index}.csv"), index=False)
