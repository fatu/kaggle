import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
from cfg import CFG


def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return


losses_dict = {'smooth-l1': nn.SmoothL1Loss,
               'mse': nn.MSELoss,
               'l1': nn.L1Loss,
               #'huber-loss': nn.HuberLoss,
               }


class Custom_Bert(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(CFG.model_path)
        config.output_hidden_states = True
        config.max_position_embeddings = CFG.max_position_embeddings
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0

        self.backbone = AutoModel.from_pretrained(CFG.model_path, config=config)

        dim = config.hidden_size

        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = 24
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.cls = nn.Sequential(
            nn.Linear(dim, CFG.num_labels)
        )
        init_params([self.cls, self.attention])

    def forward(self, input_ids, attention_mask, labels=None):
        base_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        cls_outputs = torch.stack(
            [self.dropout(layer) for layer in base_output['hidden_states'][-24:]], dim=0
        )
        cls_output = (
                torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1) * cls_outputs).sum(
            0)

        logits = torch.mean(
            torch.stack(
                [torch.sum(self.attention(self.high_dropout(cls_output)) * cls_output, dim=1) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        output = self.cls(logits)
        if labels is None:
            return output

        else:
            return (losses_dict[CFG.loss_type]()(output, labels), output)


class Custom_Bert_Simple(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(CFG.model_path)
        config.max_position_embeddings = CFG.max_position_embeddings
        config.num_labels = CFG.num_labels
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0
        self.backbone = AutoModelForSequenceClassification.from_pretrained(CFG.model_path, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        base_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        output = base_output[0]
        if labels is None:
            return output

        else:
            return (losses_dict[CFG.loss_type]()(output, labels), output)



class Custom_Bert_Mean(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(CFG.model_path)
        config.output_hidden_states = True
        config.max_position_embeddings = CFG.max_position_embeddings
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0
        self.backbone = AutoModel.from_pretrained(CFG.model_path, config=config)
        dim = config.hidden_size
        self.dropout = nn.Dropout(p=0)
        self.cls = nn.Linear(dim, CFG.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        base_output = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                )

        output = base_output.last_hidden_state
        output = self.cls(self.dropout(torch.mean(output, dim=1)))
        if labels is None:
            return output

        else:
            return (losses_dict[CFG.loss_type]()(output, labels), output)




