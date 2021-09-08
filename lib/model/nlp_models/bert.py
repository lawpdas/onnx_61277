import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import List
strList = List[str]


class BERT(nn.Module):
    def __init__(self, lr_mult=0):
        super(BERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for name, parameter in self.bert.named_parameters():
            if '11' in name:
                parameter.requires_grad_(True)
            elif '10' in name:
                parameter.requires_grad_(True)
            elif '9' in name:
                parameter.requires_grad_(True)
            elif '8' in name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)

            if not lr_mult > 0:
                parameter.requires_grad_(False)

    def forward(self, sentence: strList):

        tokens = self.tokenizer(sentence, padding='longest', return_tensors='pt')

        token_input_id = tokens['input_ids'].to(self.bert.device)
        token_type_ids = tokens['token_type_ids'].to(self.bert.device)
        token_mask = tokens['attention_mask'].to(self.bert.device)  # (N, L)

        outputs = self.bert(input_ids=token_input_id,
                            token_type_ids=token_type_ids,
                            attention_mask=token_mask,
                            output_hidden_states=True)

        # bert_out = outputs.hidden_states  # (Layers, N, L, 768)
        bert_out = outputs.last_hidden_state  # (N, L, 768)

        bert_mask = token_mask.unsqueeze(-1)  # (N, L, 1)
        bert_out = bert_out * bert_mask  # (N, L, 768)

        return bert_out, bert_mask.detach()


def build_bert_model():
    return BERT()


if __name__ == '__main__':
    bert = build_bert_model().cuda()
    # print(bert(['']).shape)
    print(bert(['', 'this is not a cat this is not a cat this is not a cat not a cat ']).shape)
