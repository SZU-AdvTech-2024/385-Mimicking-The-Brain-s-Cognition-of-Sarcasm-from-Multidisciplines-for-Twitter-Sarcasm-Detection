import torch
from torch import nn
from models.img_backbone import Backbone_img
from models.txt_backbone import Backbone_txt
from models.common import MultiheadSelfAttention, MultiheadCrossAttention


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


class Fusion(nn.Module):
    """
    fusion block
    """
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.ca12 = MultiheadCrossAttention(in_channels, out_channels)
        self.ca13 = MultiheadCrossAttention(in_channels, out_channels)
        self.ca14 = MultiheadCrossAttention(in_channels, out_channels)
        self.ca23 = MultiheadCrossAttention(in_channels, out_channels)
        self.ca24 = MultiheadCrossAttention(in_channels, out_channels)
        self.ca34 = MultiheadCrossAttention(in_channels, out_channels)

        self.ca_a = MultiheadCrossAttention(in_channels, out_channels)
        self.ca_b = MultiheadCrossAttention(in_channels, out_channels)
        self.ca_c = MultiheadCrossAttention(in_channels, out_channels)

    def forward(self, v1, v2, v3, v4):
        v12 = self.ca12(v1, v2, v2)
        v13 = self.ca13(v1, v2, v2)
        v14 = self.ca14(v1, v2, v2)
        v23 = self.ca23(v1, v2, v2)
        v24 = self.ca24(v1, v2, v2)
        v34 = self.ca34(v1, v2, v2)
        v_a = self.ca_a(v12, v34, v34)
        v_b = self.ca_a(v13, v24, v24)
        v_c = self.ca_a(v23, v14, v14)
        return v_a, v_b, v_c




class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.backbone_img = Backbone_img()
        self.backbone_txt = Backbone_txt()
        self.sa1 = MultiheadSelfAttention(in_channels, out_channels)
        self.sa2 = MultiheadSelfAttention(in_channels, out_channels)
        # self.sa3 = MultiheadSelfAttention(in_channels, out_channels)
        # self.sa4 = MultiheadSelfAttention(in_channels, out_channels)
        self.ca = MultiheadCrossAttention(in_channels, out_channels)
        # self.fusion = Fusion(in_channels, out_channels)
        self.classifier_a = Classifier(out_channels, 1)
        # self.classifier_b = Classifier(out_channels, 1)
        # self.classifier_c = Classifier(out_channels, 1)

    def forward(self, images, twts):
        with torch.no_grad():
            img_feats = self.backbone_img(images)
            twt_feats = self.backbone_txt(twts)
            # caps_feats = self.backbone_txt(caps)
            # attrs_feats = self.backbone_txt(attrs)

        v1, v2 = self.sa1(img_feats), self.sa1(twt_feats)
        v_a = self.ca(v1, v2)
        return self.classifier_a(v_a)

if __name__ == '__main__':
    model = MyModel(768, 768)

