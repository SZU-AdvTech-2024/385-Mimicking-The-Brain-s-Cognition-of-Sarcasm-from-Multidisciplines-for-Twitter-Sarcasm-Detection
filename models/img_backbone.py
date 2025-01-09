from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Backbone_img(nn.Module):
    def __init__(self, **kwargs):
        super(Backbone_img, self).__init__(**kwargs)
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # Set the model to evaluation mode
        resnet50.eval()
        layers = nn.Sequential(*list(resnet50.children())[:-1])
        self.backbone = nn.Sequential(*layers)
        # Define a transformation to normalize the input image
        # self.transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

        self.hidden = nn.Linear(2048, 1024)
        self.act = nn.ReLU()
        self.output = nn.Linear(1024, 768)

    def forward(self, x):
        # if self.transform:
        #     x = self.transform(x)
        # x = x.to(self.device)
        x = self.backbone(x).reshape(-1, 1, 2048)
        x = self.output(self.act(self.hidden(x)))

        # Print the shape of the extracted features
        # print(x.shape)
        # torch.Size([2, 1, 768])
        return x



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Backbone_img().to(device)
    image_path = '/home/p/Documents/Codes/pytorch-multimodal_sarcasm_detection/image_data/682716753374351360.jpg'

    img = Image.open(image_path)
    img = model.transform(img).to(device)
    model(img)