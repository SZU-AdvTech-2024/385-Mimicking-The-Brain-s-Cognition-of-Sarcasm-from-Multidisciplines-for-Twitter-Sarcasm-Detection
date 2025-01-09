import os
import pandas as pd
import ast
import json
from collections import namedtuple
from PIL import Image
from accelerate.test_utils.scripts.test_script import print_on
from LLMUtil import Qwen_vl


def txt2csv(txt_full_path, names):
    assert names is not None, "dataframe column names is None."
    with open(txt_full_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 处理每一行
    data = []
    for line in lines:
        arr = ast.literal_eval(line.strip())
        data.append(arr)
    df = pd.DataFrame(data, columns=names)
    df.to_csv(f"./{os.path.basename(txt_full_path).replace('txt', 'csv')}",  header=True)

def make_messages(img_full_path, item_text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "system",
                    "text":
                        """
                        You are an expert in emotional analysis and sarcasm detection. I need you to perform a multimodal sarcasm recognition task. I will provide you with an image and a piece of text. Your output should include four parts in JSON format:
                        {
                          "picture_dipict": "",
                          "text_express": "",
                          "picture_with_text_express": "",
                          "is_sarcasm": 0
                        }
                        picture_dipict: Describe what the image depicts.
                        text_express: Explain what the text expresses.
                        picture_with_text_express: State the emotion conveyed by the combination of the image and text.
                        is_sarcasm: Indicate whether the combination contains sarcasm (return 1 for yes, 0 for no).
                        Please provide detailed and clear answers for each part.
                        """
                },
                {
                    "type": "image",
                    "image": f"file://{img_full_path}",
                },
                {
                    "type": "text",
                    "text": f"{item_text}"
                },
            ],
        }
    ]
    return messages

DataEntry = namedtuple('DataEntry', ['index', 'item_idx', 'item_text', 'item_label1', 'item_label2',
                            'picture_dipict', 'text_express', 'picture_with_text_express', 'is_sarcasm', 'output_text'])

def infer_one_row(llm, idx, row):
    item_idx = row['idx']
    item_text = row['text']
    item_label1 = row['label1']
    item_label2 = row['label2']
    img_full_path = os.path.join(img_dir, str(item_idx) + '.jpg')
    messages = make_messages(img_full_path, item_text)
    output_text = llm.inference(messages)[0]
    try:
        res = json.loads(output_text)
        row_result = DataEntry(index=idx, item_idx=item_idx,
                                item_text=item_text, item_label1=item_label1,
                                item_label2=item_label2,
                                picture_dipict=res['picture_dipict'],
                                text_express=res['text_express'],
                                picture_with_text_express=res[
                                    'picture_with_text_express'],
                                is_sarcasm=res['is_sarcasm'], output_text=None)
    except json.JSONDecodeError:
        row_result = DataEntry(index=idx, item_idx=item_idx,
                                item_text=item_text, item_label1=item_label1,
                                item_label2=item_label2,
                               picture_dipict=None,
                               text_express=None,
                               picture_with_text_express=None,
                               is_sarcasm=None,
                               output_text=output_text)
    finally:
        return row_result



def main(llm, csv_name, img_dir):
    df = pd.read_csv(csv_name)
    for idx, row in df.iterrows():
        row_result = infer_one_row(llm, idx, row)

if __name__ == "__main__":
    txt_dir = "/home/p/Documents/Codes/my_Qwen2vl_eval/dataset/data-of-multimodal-sarcasm-detection/text/"
    txt_name = "test2.txt"
    img_dir = "/home/p/Documents/Codes/my_Qwen2vl_eval/dataset/dataset_image/"

    # txt2csv(txt_dir + txt_name, names=["idx", "text", "label1", "label2"])
    llm = Qwen_vl(pretrained_model_name="Qwen/Qwen2-VL-2B-Instruct")
    main(llm, csv_name="./test2.csv", img_dir=img_dir)
