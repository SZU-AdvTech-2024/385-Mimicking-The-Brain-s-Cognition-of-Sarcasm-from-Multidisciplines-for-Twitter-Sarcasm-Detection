import torch
from torch import nn
import torch.nn.functional as F
from models.img_backbone import Backbone_img
from models.txt_backbone import Backbone_txt
from models.common import MultiheadSelfAttention, MultiheadCrossAttention

class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerSALayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerCALayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerCALayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_2, src_3, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.cross_attn(src, src_2, src_3, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

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
        self.ca12 = TransformerCALayer(in_channels, 8, dim_feedforward=in_channels, dropout=0.1)
        # self.ca13 = MultiheadCrossAttention(in_channels, out_channels)
        # self.ca14 = MultiheadCrossAttention(in_channels, out_channels)
        # self.ca23 = MultiheadCrossAttention(in_channels, out_channels)
        # self.ca24 = MultiheadCrossAttention(in_channels, out_channels)
        # self.ca34 = MultiheadCrossAttention(in_channels, out_channels)
        #
        # self.ca_a = MultiheadCrossAttention(in_channels, out_channels)
        # self.ca_b = MultiheadCrossAttention(in_channels, out_channels)
        # self.ca_c = MultiheadCrossAttention(in_channels, out_channels)

    def forward(self, v1, v2):
        v12 = self.ca12(v1, v2, v2)
        # v13 = self.ca13(v1, v3, v3)
        # v14 = self.ca14(v1, v4, v4)
        # v23 = self.ca23(v2, v3, v3)
        # v24 = self.ca24(v2, v4, v4)
        # v34 = self.ca34(v3, v4, v4)
        # v_a = self.ca_a(v12, v34, v34)
        # v_b = self.ca_b(v13, v24, v24)
        # v_c = self.ca_c(v23, v14, v14)
        return v12# , v_b, v_c




class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.backbone_img = Backbone_img()
        self.backbone_txt = Backbone_txt()
        self.sa1 = TransformerSALayer(in_channels, 8, dim_feedforward=in_channels, dropout=0.1)
        self.sa2 = TransformerSALayer(in_channels, 8, dim_feedforward=in_channels, dropout=0.1)

        self.fusion = Fusion(in_channels, out_channels)
        self.classifier_a = Classifier(out_channels, 1)
        # self.classifier_b = Classifier(out_channels, 1)
        # self.classifier_c = Classifier(out_channels, 1)

    def forward(self, images, twts, caps, attrs):
        with torch.no_grad():
            img_feats = self.backbone_img(images)
            twt_feats = self.backbone_txt(twts)
            # caps_feats = self.backbone_txt(caps)
            # attrs_feats = self.backbone_txt(attrs)

        v1, v2 = self.sa1(img_feats), self.sa2(twt_feats)
        v_a = self.fusion(v1, v2)

        return v1, v2, self.classifier_a(v_a)

if __name__ == '__main__':
    model = MyModel(768, 768)

