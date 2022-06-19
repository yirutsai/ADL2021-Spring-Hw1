from audioop import bias
from socket import PACKET_LOOPBACK
from typing import Dict
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        netType:str,
        pack:bool,
        attn,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional  = bidirectional
        self.num_class = num_class
        self.pack = pack
        self.attn = attn
        self.netType = netType
        self.dropoutLayer = nn.Dropout(dropout)
        # TODO: model architecture
        if(netType=="RNN"):
            self.net = nn.RNN(input_size=embeddings.shape[1],hidden_size = self.hidden_size,num_layers = self.num_layers,dropout = self.dropout,bidirectional=self.bidirectional,batch_first=True)
        elif(netType=="LSTM"):
            self.net = nn.LSTM(input_size=embeddings.shape[1],hidden_size = self.hidden_size,num_layers = self.num_layers,dropout = self.dropout,bidirectional=self.bidirectional,batch_first=True)
        elif(netType=="GRU"):
            self.net = nn.GRU(input_size=embeddings.shape[1],hidden_size = self.hidden_size,num_layers = self.num_layers,dropout = self.dropout,bidirectional=self.bidirectional,batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_output_size,self.num_class)
        )
        self.attention = nn.MultiheadAttention(self.encoder_output_size,1,dropout=self.dropout)
        
    def attn_net(self,encoder_out,hidden):
        # hidden = hidden.squeeze(0)
        # print(f"encoder_out.shape:{encoder_out.shape}")                 #(batch_size,80,1024)
        # print(f"hidden.shape:{hidden.unsqueeze(2).shape}")              #(batch_size,1024,1)
        attn_w = torch.bmm(encoder_out,hidden.unsqueeze(2)).squeeze(2)
        # print(f"attn_w.shape:{attn_w.shape}")
        soft_attn_w = F.softmax(attn_w,1)
        # print(f"soft_attn_w.shape:{soft_attn_w.shape}")                 #(batch_size,80)
        hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_w.unsqueeze(2))     #(batch_size,1024)

        # hidden = torch.bmm(encoder_out,hidden)
        # hidden = F.softmax(hidden,1)
        # hidden = torch.bmm(encoder_out.transpose(1,2),hidden)
        return hidden.squeeze(2)
    @property
    def encoder_output_size(self) -> int:
        if(self.bidirectional):
            return self.hidden_size*2
        else:
            return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x, y = batch["text"],batch["intent"]            # (batch_size, max_len), (batch_size)

        x = self.embed(x)                               # (batch_size, max_len, 300)
        x = self.dropoutLayer(x)
        if(self.pack):
            x_packed = pack_padded_sequence(x,batch["length"],batch_first= True,enforce_sorted=False)
        # print(x.shape)
            if(self.netType!="LSTM"):
                out_packed,hn = self.net(x_packed)
            else:
                out_packed, (hn,cn) = self.net(x_packed)
            self.net.flatten_parameters()
            out, _ =  pad_packed_sequence(out_packed,batch_first=True)      #(batch_size,80,1024)
        else:
            if(self.netType!="LSTM"):
                out,hn = self.net(x)
            else:
                out,(hn,cn) = self.net(x)
            self.net.flatten_parameters()
        
        # print(f"out.shape:{out.shape}")
        # out = out[:,:,self.hidden_size:] + out[:,:,:self.hidden_size]
        # print(f"out.shape:{out.shape}")
        # print(hn.shape)
        if(self.bidirectional):
            hn = torch.cat((hn[-1],hn[-2]),axis = 1)
        else:
            hn = hn[-1]
        # print(hn.shape)
        # print(out.shape)
        # print(f"hn.shape:{hn.shape}")                                   # (batch_size,1024)
        if(self.attn):
            atten_out = self.attn_net(out,hn)
            logits = self.fc_layers(atten_out)
        else:
            logits = self.fc_layers(hn)
        # print(logits.shape)
        return logits

class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        cnn_num_layers:int,
        dropout: float,
        bidirectional: bool,
        netType:str,
        num_class:int,
        pack:bool,
        max_len:int,
        use_crf:bool,
    ) -> None:
        super(SeqTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn_num_layers = cnn_num_layers
        self.dropout = dropout
        self.bidirectional  = bidirectional
        self.num_class= num_class
        self.pack = pack
        self.max_len = max_len
        self.use_crf = use_crf
        self.eval_mode= False
        self.dropoutLayer = nn.Dropout(dropout)
        # TODO: model architecture
        self.cnn = nn.Sequential(
                nn.Conv1d(self.max_len, self.max_len, 5, 1, 2),
                nn.ReLU(),
                nn.Dropout()
        )
        if(netType=="RNN"):
            self.net = nn.RNN(input_size=embeddings.shape[1],hidden_size = self.hidden_size,num_layers = self.num_layers,dropout = self.dropout,bidirectional=self.bidirectional,batch_first=True)
        elif(netType=="LSTM"):
            self.net = nn.LSTM(input_size=embeddings.shape[1],hidden_size = self.hidden_size,num_layers = self.num_layers,dropout = self.dropout,bidirectional=self.bidirectional,batch_first=True)
        elif(netType=="GRU"):
            self.net = nn.GRU(input_size=embeddings.shape[1],hidden_size=hidden_size,num_layers=num_layers,bidirectional=self.bidirectional,dropout=self.dropout,batch_first = True)
        self.fc_layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.encoder_output_size, self.num_class)
        )
        if(self.use_crf):
            self.crf = CRF(self.encoder_output_size, self.num_class)
        
    @property
    def encoder_output_size(self) -> int:
        if(self.bidirectional):
            return self.hidden_size*2
        else:
            return self.hidden_size
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        outputDict = {}
        x, y = batch["tokens"],batch["tags"]            # (batch_size, max_len), (batch_size)
        mask = batch["mask"]
        # print(f"x.shape:{x.shape}")
        x = self.embed(x)                               # (batch_size, max_len, 300)
        x = self.dropoutLayer(x)
        for _ in range(self.cnn_num_layers):
            x = self.cnn(x)
        # print(f"x.shape:{x.shape}")
        if(self.pack):
            x_packed = pack_padded_sequence(x,batch["length"],batch_first= True,enforce_sorted=False)
            out_packed, _ = self.net(x_packed)
            out, _ =  pad_packed_sequence(out_packed,batch_first=True,total_length=self.max_len)      #(batch_size,80,1024)
        else:
            out,_ = self.net(x)
        # print(f"out.shape:{out.shape}")
        if(self.use_crf):
            outputDict['loss'] = self.crf.loss(out, y, mask)
            outputDict["max_score"],outputDict["pred_labels"] = self.crf(out,mask)
        else:
            logits= self.fc_layers(out)
            if(self.eval_mode== False):
                outputDict["loss"] = F.cross_entropy(logits.permute(0,2,1), y[:,:logits.shape[1]],ignore_index=10)

            outputDict["pred_labels"] =logits.detach().cpu().max(2,keepdim = True).indices.squeeze(2)

        return outputDict

class CRF(nn.Module):
    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags + 3
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = -1e4
        self.transitions.data[:, self.stop_idx] = -1e4
    def log_sum_exp(self,x):
        max_score = x.max(-1)[0]
        return max_score + torch.log(torch.sum(torch.exp(x - max_score.unsqueeze(-1)), -1))
    def forward(self, x, mask):
        x = self.fc(x)
        return self._viterbi_decode(x, mask)
    def loss(self, x, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        x = self.fc(x)

        L = x.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(x, masks_)
        gold_score = self._score_sentence(x, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss
    def _score_sentence(self, x, tags, mask):
        B, L, C = x.shape
        seq_len = mask.sum(-1).long() # [B]

        emit_scores = x.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1) # [B, L]
        
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long,device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1) # [B, L + 1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]] # [B, L]

        last_tag = tags.gather(dim=1, index=seq_len.unsqueeze(-1)).squeeze(-1)
        last_score = self.transitions[self.stop_idx, last_tag] # [B]

        score = ((emit_scores + trans_scores) * mask).sum(1) + last_score

        return score
    
    def _viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        # bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers
        bps = torch.zeros_like(features).long()

        # Initialize the viterbi variables in log space
        # max_score = torch.full((B, C), -1e4, device=features.device)  # [B, C]
        max_score = torch.full_like(features[:, 0, :], -1e4)
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            # seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            # for bps_t in reversed(bps[b, :seq_len]):
            for bps_t in reversed(bps[b]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])
        best_paths = np.array(best_paths,dtype=np.int8)
        return best_score, best_paths
    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), -1e4, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = self.log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = self.log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores