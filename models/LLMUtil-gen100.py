import tqdm
from sympy import pretty
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json




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
    llm = Qwen_vl()
    import os
    dir_path = '/home/p/Documents/Codes/my_Qwen2vl_eval/dataset/dataset_image'
    file = open('/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection/text_data/train.txt', "rb")
    names = []
    for line in file:
        if len(names) == 100:
            break
        content = eval(line)
        image = content[0]
        sentence = content[1]
        group = content[2]
        if os.path.isfile(
                os.path.join(dir_path, image + ".jpg")):
            names.append(image)

    # # 获取文件夹下所有文件名
    # file_names = [f for f in os.listdir(dir_path) if
    #               os.path.isfile(os.path.join(dir_path, f))]

    result = {}
    for file_name in tqdm.tqdm(names, total=100):
        file_name = file_name + ".jpg"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{dir_path}/{file_name}",
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ],
            }
        ]
        output_text = llm.inference(messages)[0]
        result[file_name.split('.')[0]] = output_text
    json.dump(result, open("./img2txt_Qwen2VL_100.json", "w"), indent=4)
    print('done!')

