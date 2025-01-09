
数据集地址下载：
https://github.com/headacheboy/data-of-multimodal-sarcasm-detection


数据集图像软链接到项目：
```bash
ln -s /home/p/Documents/Datasets/data-of-multimodal-sarcasm-detection/dataset_image/ .
```

程序运行：
```bash
conda activate llm
cd /home/p/Documents/Codes/SarcasmDetection

python main.py \
  --modality itc \
  --fus weighted \
  --batch-size 256
```
