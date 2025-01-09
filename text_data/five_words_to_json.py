import os, json
import tqdm


"""
# txt to json
with open('/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection_V2/text_data/img_to_five_words.txt', "rb") as f:
    lines = f.readlines()

result = {}
for idx, line in tqdm.tqdm(enumerate(lines), total=len(lines)):
    item = {}
    content = eval(line)
    key = content[0]
    words = ', '.join(content[1:])
    result[key] = words

json.dump(result, open("./img_to_five_words.json", "w"), indent=4)
"""



# test if json cover all key in train and test dataset
with open('/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection_V2/text_data/img_to_five_words.json', 'r', encoding='utf-8') as attr_f:
    attrs = json.load(attr_f)

with open('/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection_V2/text_data/caption_train.json', 'r', encoding='utf-8') as train_f:
    train = json.load(train_f)

ks = []
for k in attrs.keys():
    ks.append(k)
for k in train.keys():
    if k not in ks:
        print(k)
# OK