import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json


class MyDataset(Dataset):
    def __init__(self, cap_file_name, twt_file_name, attr_file_name, img_data_dir, transform=None):
        """
        初始化数据集
        Args:
            data_dir (str): 数据所在目录
            transform (callable, optional): 数据预处理/变换操作
            image, caption, twt
        """

        # 读取 JSON 文件
        with open(cap_file_name, 'r', encoding='utf-8') as cap_file:
            caps = json.load(cap_file)

        with open(attr_file_name, 'r', encoding='utf-8') as attr_file:
            attrs = json.load(attr_file)

        self.data_dict = {}
        with open(twt_file_name, "rb") as twt_file:
            for line in twt_file:
                # if len(self.data_dict) == 100:
                #     break
                content = eval(line)
                key = content[0]
                twt = content[1]
                if 'test' in os.path.basename(twt_file_name):
                    label = content[3]
                else:
                    label = content[2]
                img_path = os.path.join(img_data_dir, key + ".jpg")
                if os.path.isfile(img_path) and key in caps: # if image exists and caption exists
                    self.data_dict[key] = {"twt": twt, "cap": caps[key], 'attr': attrs[key], 'img_path': img_path, "label": label}
        self.transform = transform
        self.data = list(self.data_dict.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取数据
        Args:
            idx (int): 数据索引
        Returns:
            data: 处理后的数据
            label: 数据标签（如果有的话）
        """
        key = self.data[idx]
        item = self.data_dict[key]
        twt, cap, attr, img_path, label = item["twt"], item["cap"], item[
            "attr"], item["img_path"], item["label"]
        image = Image.open(img_path).convert('RGB')  # 加载图像并转换为RGB格式

        if self.transform:
            image = self.transform(image)  # 应用变换

        return image, twt, cap, attr, label

if __name__ == '__main__':
    # 使用示例
    twt_data_dir = '/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection/text_data/'
    img_data_dir = '/home/p/Documents/Datasets/data-of-multimodal-sarcasm-detection/dataset_image/'
    cap_data_dir = '/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection/text_data/'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 标准化
    ])

    dataset = MyDataset(twt_data_dir=twt_data_dir, img_data_dir=img_data_dir, cap_data_dir=cap_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 测试读取数据
    for images, twts, caps, labels in dataloader:
        print(images.shape, twts, caps, labels)
        break
