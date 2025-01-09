import torch
from torch import nn
from .img_backbone import Backbone_img
from .txt_backbone import Backbone_txt
from .common import MultiheadSelfAttention, MultiheadCrossAttention


class Block(nn.Module):
    """
    attention block
    """
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.sa1 = MultiheadSelfAttention(in_channels, out_channels)
        self.sa2 = MultiheadSelfAttention(in_channels, out_channels)
        self.sa3 = MultiheadSelfAttention(in_channels, out_channels)
        self.ca1 = MultiheadCrossAttention(in_channels, out_channels)
        self.ca2 = MultiheadCrossAttention(in_channels, out_channels)


    def forward(self, x, y, z):
        x = self.sa1(x)
        y = self.sa2(y)
        z = self.sa3(z)
        x = self.ca1(x, y, y)
        z = self.ca2(z, y, y)
        return x, y, z


class Fusion_block(nn.Module):
    """
    fusion block
    """
    def __init__(self, in_channels, out_channels):
        super(Fusion_block, self).__init__()
        self.ca = MultiheadCrossAttention(in_channels, out_channels)

    def forward(self, x, y, z):
        return self.ca(x, z, z)

class Classifier(nn.Module):
    """
    classifier block
    """
    def __init__(self, input_size, out_channels):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, out_channels)  # 输出层，二分类用单一输出节点

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 使用 Sigmoid 激活
        return x.squeeze()




class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_block=1, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.backbone_img = Backbone_img()
        self.backbone_txt = Backbone_txt()

        # self.blocks = nn.Sequential(*[Block(in_channels, out_channels) for _ in range(n_block)])
        self.blocks = nn.ModuleList(
            [Block(in_channels, out_channels) for _ in range(n_block)])
        self.fusion_block = Fusion_block(in_channels, out_channels)
        self.classifier = Classifier(out_channels, 1)

    def forward(self, images, twts, caps):
        with torch.no_grad():
            img_feats = self.backbone_img(images)
            twt_feats = self.backbone_txt(twts)
            caps_feats = self.backbone_txt(caps)

        # x, y, z = self.blocks((img_feats, twt_feats, caps_feats))
        x, y, z = img_feats, twt_feats, caps_feats
        for block in self.blocks:
            x, y, z = block(x, y, z)
        f = self.fusion_block(x, y, z)
        return self.classifier(f)

if __name__ == '__main__':
    model = MyModel(768, 768)

