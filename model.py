import torch
from torch import nn
import torch.nn.functional as F
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
        # self.ca21 = MultiheadCrossAttention(in_channels, out_channels)
        # self.ca1221 = MultiheadCrossAttention(in_channels, out_channels)
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
        return v1, v2, v12 # , v_b, v_c


class WeightedFusion_mod3(nn.Module):
    def __init__(self, feature_dim):
        super(WeightedFusion_mod3, self).__init__()
        print('fusion: WeightedFusion_mod3')
        # 定义一个小型 MLP，用于生成注意力权重
        self.attention_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 3)  # 输出两个权重
        )

    def forward(self, img_feat, text_feat, x_feat):
        img_feat, text_feat, x_feat = img_feat.squeeze(), text_feat.squeeze(), x_feat.squeeze()  # 1. 拼接特征
        concat_feat = torch.cat([img_feat, text_feat, x_feat], dim=-1)

        # 2. 生成注意力权重，并通过 softmax 归一化
        weights = F.softmax(self.attention_mlp(concat_feat), dim=-1)

        w_img, w_text, w_x = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        w_img = w_img.unsqueeze(0)  # 使其变成 (1, 2)
        w_text = w_text.unsqueeze(0)
        w_x = w_x.unsqueeze(0)
        # 3. 计算加权融合特征
        fused_feat = w_img * img_feat + w_text * text_feat + w_x * x_feat
        return img_feat, text_feat, x_feat, fused_feat.squeeze()


class WeightedFusion_mod4(nn.Module):
    def __init__(self, feature_dim):
        super(WeightedFusion_mod4, self).__init__()
        print('fusion: WeightedFusion_mod4')
        # 定义一个小型 MLP，用于生成注意力权重
        self.attention_mlp = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 4)  # 输出两个权重
        )

    def forward(self, img_feat, text_feat, x_feat, y_feat):
        img_feat, text_feat, x_feat, y_feat = img_feat.squeeze(), text_feat.squeeze(), x_feat.squeeze(), y_feat.squeeze()  # 1. 拼接特征
        concat_feat = torch.cat([img_feat, text_feat, x_feat, y_feat], dim=-1)

        # 2. 生成注意力权重，并通过 softmax 归一化
        weights = F.softmax(self.attention_mlp(concat_feat), dim=-1)

        w_img, w_text, w_x, w_y = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3], weights[:, 3:4]
        w_img = w_img.unsqueeze(0)  # 使其变成 (1, 2)
        w_text = w_text.unsqueeze(0)
        w_x = w_x.unsqueeze(0)
        w_y = w_y.unsqueeze(0)
        # 3. 计算加权融合特征
        fused_feat = w_img * img_feat + w_text * text_feat + w_x * x_feat + w_y * y_feat
        return img_feat, text_feat, x_feat, y_feat,  fused_feat.squeeze()


class WeightedFusion_mod2(nn.Module):
    def __init__(self, feature_dim):
        super(WeightedFusion_mod2, self).__init__()
        print('fusion: WeightedFusion_mod2')
        # 定义一个小型 MLP，用于生成注意力权重
        self.attention_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2)  # 输出两个权重
        )

    def forward(self, img_feat, text_feat):
        img_feat, text_feat=img_feat.squeeze(), text_feat.squeeze()# 1. 拼接特征
        concat_feat = torch.cat([img_feat, text_feat], dim=-1)

        # 2. 生成注意力权重，并通过 softmax 归一化
        weights = F.softmax(self.attention_mlp(concat_feat), dim=-1)

        w_img, w_text = weights[:, 0:1], weights[:, 1:2]
        w_img = w_img.unsqueeze(0)  # 使其变成 (1, 2)
        w_text = w_text.unsqueeze(0)
        # 3. 计算加权融合特征
        fused_feat = w_img * img_feat + w_text * text_feat
        return img_feat, text_feat, fused_feat.squeeze()


class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, opt, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.backbone_img = Backbone_img()
        self.backbone_txt = Backbone_txt()
        self.sa1 = MultiheadSelfAttention(in_channels, out_channels)
        self.sa2 = MultiheadSelfAttention(in_channels, out_channels)
        self.sa3 = MultiheadSelfAttention(in_channels, out_channels)
        self.sa4 = MultiheadSelfAttention(in_channels, out_channels)

        # self.fusion = Fusion(in_channels, out_channels)
        n_mod = len(opt.modality)
        if opt.fus == 'weighted':
            if n_mod == 2:
                self.fusion = WeightedFusion_mod2(in_channels)
            elif n_mod == 3:
                self.fusion = WeightedFusion_mod3(in_channels)
            elif n_mod == 4:
                self.fusion = WeightedFusion_mod4(in_channels)
        elif opt.fus == 'attn':
            pass
        self.classifier_a = Classifier(out_channels, 1)
        # self.classifier_b = Classifier(out_channels, 1)
        # self.classifier_c = Classifier(out_channels, 1)

    def handle_itca(self, images, twts, caps, attrs):
        with torch.no_grad():
            img_feats = self.backbone_img(images)
            twt_feats = self.backbone_txt(twts)
            caps_feats = self.backbone_txt(caps)
            attrs_feats = self.backbone_txt(attrs)

        v1, v2, v3, v4 = self.sa1(img_feats), self.sa2(twt_feats), self.sa3(caps_feats), self.sa4(attrs_feats)
        v1, v2, v3, v4, v_a = self.fusion(v1, v2, v3, v4)
        return (v1, v2, v3, v4), self.classifier_a(v_a)

    def handle_ct(self, twts, caps):
        with torch.no_grad():
            twt_feats = self.backbone_txt(twts)
            caps_feats = self.backbone_txt(caps)

        v1, v2 = self.sa1(twt_feats), self.sa2(caps_feats)
        v1, v2, v_a = self.fusion(v1, v2)
        return (v1, v2), self.classifier_a(v_a)

    def handle_ic(self, images, caps):
        pass
    def handle_it(self, images, twts):
        pass
    def handle_itc(self, images, twts, caps):
        with torch.no_grad():
            img_feats = self.backbone_img(images)
            twt_feats = self.backbone_txt(twts)
            caps_feats = self.backbone_txt(caps)
            # attrs_feats = self.backbone_txt(attrs)

        v1, v2, v3 = self.sa1(img_feats), self.sa2(twt_feats), self.sa3(caps_feats)#, self.sa4(attrs_feats)
        v1, v2, v3, v_a = self.fusion(v1, v2, v3)
        return (v1, v2, v3), self.classifier_a(v_a)
        
    def forward(self, images, twts, caps, attrs):
        # all modality
        if images is not None and twts is not None and caps is not None and attrs is not None:
            return self.handle_itca(images, twts, caps, attrs)
        # cap_twt
        elif images is None and twts is not None and caps is not None and attrs is None:
            return self.handle_ct(twts, caps)
        # img cap
        elif images is not None and twts is None and caps is not None and attrs is None:
            return self.handle_ic(images, caps)
        # img twt
        elif images is not None and twts is not None and caps is  None and attrs is None:
            return self.handle_it(images, twts)
        # img twt cap
        elif images is not None and twts is not None and caps is not None and attrs is None:
            return self.handle_itc(images, twts, caps)
        else:
            exit(1)



if __name__ == '__main__':
    model = MyModel(768, 768)

