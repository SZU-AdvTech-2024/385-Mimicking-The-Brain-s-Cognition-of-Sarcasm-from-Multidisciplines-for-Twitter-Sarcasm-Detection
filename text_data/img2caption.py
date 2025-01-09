import os
import time, datetime
import tqdm, argparse
from sympy import pretty
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import sys
sys.path.append('../')
from utils.logutil import LoggerConfig, create_logger



class Qwen_vl():
    def __init__(self, pretrained_model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.pretrained_model_name = pretrained_model_name
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, torch_dtype="auto", device_map="balanced_low_0"
        )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-2B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )
        # default processer
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    def inference(self, messages):
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=12800)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--twt_data_dir', type=str,
                        default='/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection/text_data/',
                        help='twt_data_dir')
    parser.add_argument('--img_data_dir', type=str,
                        default='/home/p/Documents/Datasets/data-of-multimodal-sarcasm-detection/dataset_image/',
                        help='img_data_dir')
    parser.add_argument('--cap_data_dir', type=str,
                        default='/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection/text_data/',
                        help='cap_data_dir')
    parser.add_argument('--data_type', type=str,
                        default='train', help='data_type train/valid/test')
    parser.add_argument('--begin', type=int, default=1)
    parser.add_argument('--end', type=int, default=1000)

    opt = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), opt.data_type)
    logger_config = LoggerConfig(opt.data_type, log_file_name=f'{opt.begin}-{opt.end}.txt')
    logger = create_logger(__file__, logger_config)


    llm = Qwen_vl()
    import os

    # result = {}
    if opt.data_type == 'train':
        txt_name = 'train.txt'
    elif opt.data_type == 'valid':
        txt_name = 'valid.txt'
    elif opt.data_type == 'test':
        txt_name = 'test.txt'
    else:
        exit(1)
    with open(os.path.join(opt.twt_data_dir, txt_name), "rb") as twt_file:
        lines = twt_file.readlines()
    group = lines[opt.begin-1:opt.end]
    for idx, line in tqdm.tqdm(enumerate(group), total=len(group)):
        item = {}
        content = eval(line)
        key = content[0]
        twt = content[1]
        label = content[2]
        img_path = os.path.join(opt.img_data_dir, key + ".jpg")
        if os.path.isfile(
                img_path):  # if image exists
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{img_path}",
                        },
                        {
                            "type": "text",
                            "text": "Describe this image."
                        }
                    ],
                }
            ]
            output_text = llm.inference(messages)[0]
        else:
            output_text = None
        # result[key] = output_text
        item['idx'] = idx + 1
        item['key'] = key
        item['caption'] = output_text
        string = json.dumps(item)
        logger.info(string)
        localtime = time.asctime( time.localtime(time.time()) )
        print(f'{localtime}: {string}')
    print('done!')




