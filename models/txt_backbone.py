import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


class Backbone_txt(nn.Module):
    def __init__(self, model_name="roberta-base", **kwargs):
        super(Backbone_txt, self).__init__(**kwargs)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

    def forward(self, x):
        _input = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        _input.to('cuda')
        with torch.no_grad():
            outputs = self.model(**_input)
        last_hidden_state = outputs.last_hidden_state  #  torch.Size([b, 14, 768])
        sentence_embedding = torch.mean(last_hidden_state, dim=1)  #  torch.Size([2, 768])
        sentence_embedding = sentence_embedding.unsqueeze(1)  #  torch.Size([2, 1, 768])
        return sentence_embedding

if __name__ == '__main__':
    text = "RoBERTa is a robustly optimized BERT model."
    text = [text for i in range(3)]
    model = Backbone_txt()
    ret = model(text)
    print(ret.shape)
