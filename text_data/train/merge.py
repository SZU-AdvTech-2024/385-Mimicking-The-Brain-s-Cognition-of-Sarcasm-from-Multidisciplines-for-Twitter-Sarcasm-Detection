import os
import sys
import json


all_caps = {}
no_cap = []
# {"idx": 2, "key": "725333760762363905", "caption": null}
def merge_one(fname):
    with open(os.path.join(fname, fname), "rb") as twt_file:
        lines = twt_file.readlines()
    for line in lines:
        j_line = json.loads(line)
        # key = j_line['idx']
        key = j_line['key']
        cap = j_line['caption']
        if not cap:
            no_cap.append(key)
            continue
        if key not in all_caps:
            all_caps[key] = cap

if __name__ == '__main__':
    target_dir = '/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection/text_data/train'
    fnames = ['1-10000.txt', '10000-15000.txt', '15000-20000.txt', '20000-23000.txt', '23000-26000.txt', '26000-30000.txt']
    for fname in fnames:
        merge_one(os.path.join(target_dir, fname))

    json.dump(all_caps, open(os.path.join(target_dir, "caption_train.json"), "w"), indent=4)
    print(f'len(all_caps) = {len(all_caps)}')
    print(f'len(no_cap) = {len(no_cap)}')
    with open(os.path.join(target_dir, "no_cap.txt"), "w") as file:
        str_no_cap = '\n'.join(no_cap)
        file.write(str_no_cap)  # 写入文本








