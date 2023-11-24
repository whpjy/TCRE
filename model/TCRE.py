import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TCRE_unit(nn.Module):
    def __init__(self, args, input_size):
        super(TCRE_unit, self).__init__()
        self.args = args

        self.att1 = nn.MultiheadAttention(300, 2)
        self.att2 = nn.MultiheadAttention(300, 2)
        self.input_transform1 = nn.Linear(input_size, args.hidden_size * 3, bias=True)
        self.input_transform2 = nn.Linear(input_size, args.hidden_size * 3, bias=True)
        self.input_transform3 = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=True)
        self.input_transform4 = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=True)
        self.weight1 = nn.Parameter(torch.ones(args.hidden_size * 2, 2), requires_grad=True)
        self.weight2 = nn.Parameter(torch.ones(args.hidden_size * 2, 2), requires_grad=True)

    def forward(self, x):

        gates1 = self.input_transform1(x)
        gates2 = self.input_transform2(x)
        c_e_current, g_e_adapt, g_e_inter = gates1[:, :, :].chunk(3, 2)
        c_r_current, g_r_adapt, g_r_inter = gates2[:, :, :].chunk(3, 2)

        c_e_current = torch.tanh(c_e_current)
        g_e_adapt = torch.sigmoid(g_e_adapt)
        h_e_adapt = g_e_adapt * c_e_current

        c_r_current = torch.tanh(c_r_current)
        g_r_adapt = torch.sigmoid(g_r_adapt)
        h_r_adapt = g_r_adapt * c_r_current

        g_e_inter = torch.sigmoid(g_e_inter)
        h_inter_e_by_gate = g_e_inter * h_r_adapt
        h_inter_e_by_att = self.att1(c_e_current, c_e_current, c_r_current)[0]
        h_inter_e = self.input_transform3(torch.cat((h_inter_e_by_gate, h_inter_e_by_att), dim=-1))
        i_e = torch.cat((h_e_adapt,h_inter_e), dim=-1)

        g_r_inter = torch.sigmoid(g_r_inter)
        h_inter_r_by_gate = g_r_inter * h_e_adapt
        h_inter_r_by_att = self.att2(c_r_current, c_r_current, c_e_current)[0]
        h_inter_r = self.input_transform4(torch.cat((h_inter_r_by_gate, h_inter_r_by_att), dim=-1))
        i_r = torch.cat((h_r_adapt, h_inter_r), dim=-1)

        f1, f2 = F.softmax(i_e @ self.weight1, dim=-1).chunk(2, dim=-1)
        f3, f4 = F.softmax(i_r @ self.weight2, dim=-1).chunk(2, dim=-1)
        h_e = f1.expand_as(i_e) * i_e + f2.expand_as(i_r) * i_r
        h_r = f3.expand_as(i_e) * i_e + f4.expand_as(i_r) * i_r
        loss_f = abs(math.log(torch.sum(f1 * f4, dim=2).sum().item())/ (f1.shape[0] * f1.shape[1]))

        h_e = torch.cat((i_e, h_e), dim=-1)
        h_r = torch.cat((i_r, h_r), dim=-1)

        return h_e, h_r, loss_f


class encoder(nn.Module):
    def __init__(self, args, input_size):
        super(encoder, self).__init__()
        self.args = args
        self.unit = TCRE_unit(args, input_size)

    def forward(self, x):
        h_ner, h_re, loss_f = self.unit(x)
        return h_ner, h_re, loss_f


class ner_unit(nn.Module):
    def __init__(self, args, ner2idx):
        super(ner_unit, self).__init__()
        self.hidden_size = args.hidden_size
        self.ner2idx = ner2idx
        self.hid2hid = nn.Linear(self.hidden_size * 12, self.hidden_size)
        self.hid2tag = nn.Linear(self.hidden_size, len(ner2idx))
        self.batch_size = args.batch_size
        self.elu = nn.ELU()
        self.ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, h_ner, mask):
        length, batch_size, _ = h_ner.size()

        h_global = torch.max(h_ner, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        st = h_ner.unsqueeze(1).repeat(1, length, 1, 1)
        en = h_ner.unsqueeze(0).repeat(length, 1, 1, 1)
        ner = torch.cat((st, en, h_global), dim=-1)
        ner = self.ln(self.hid2hid(ner))
        ner = self.elu(self.dropout(ner))
        ner = torch.sigmoid(self.hid2tag(ner))

        diagonal_mask = torch.triu(torch.ones(batch_size, length, length)).to(device)
        diagonal_mask = diagonal_mask.permute(1, 2, 0)
        mask_s = mask.unsqueeze(1).repeat(1, length, 1)
        mask_e = mask.unsqueeze(0).repeat(length, 1, 1)

        mask_ner = mask_s * mask_e
        mask = diagonal_mask * mask_ner
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, len(self.ner2idx))
        ner = ner * mask
        return ner


class re_unit(nn.Module):
    def __init__(self, args, re2idx):
        super(re_unit, self).__init__()
        self.hidden_size = args.hidden_size
        self.relation_size = len(re2idx)
        self.re2idx = re2idx
        self.hid2hid = nn.Linear(self.hidden_size * 12, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, self.relation_size)
        self.elu = nn.ELU()
        self.batch_size = args.batch_size
        self.r = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, h_re, mask):
        length, batch_size, _ = h_re.size()

        h_global = torch.max(h_re, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        r1 = h_re.unsqueeze(1).repeat(1, length, 1, 1)
        r2 = h_re.unsqueeze(0).repeat(length, 1, 1, 1)
        re = torch.cat((r1, r2, h_global), dim=-1)

        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = torch.sigmoid(self.hid2rel(re))

        mask = mask.unsqueeze(-1).repeat(1, 1, self.relation_size)
        mask_e1 = mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask_e2 = mask.unsqueeze(0).repeat(length, 1, 1, 1)
        mask = mask_e1 * mask_e2
        re = re * mask

        return re


class TCRE(nn.Module):
    def __init__(self, args, input_size, ner2idx, rel2idx):
        super(TCRE, self).__init__()
        self.args = args
        self.feature_extractor = encoder(args, input_size)

        self.ner = ner_unit(args, ner2idx)
        self.re = re_unit(args, rel2idx)
        self.dropout = nn.Dropout(args.dropout)

        if args.embed_mode == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained("./pretrain/albert")
            self.bert = AlbertModel.from_pretrained("./pretrain/albert")
        elif args.embed_mode == 'bert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.bert = AutoModel.from_pretrained("bert-base-cased")
        elif args.embed_mode == 'scibert':
            self.tokenizer = AutoTokenizer.from_pretrained("pretrain/scibert")
            self.bert = AutoModel.from_pretrained("./pretrain/scibert")

    def forward(self, x, mask):

        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           is_split_into_words=True).to(device)
        x = self.bert(**x)[0]

        x = x.transpose(0, 1)

        h_ner, h_re, loss_f = self.feature_extractor(x)

        ner_score = self.ner(h_ner, mask)
        re_core = self.re(h_re, mask)
        return ner_score, re_core, loss_f