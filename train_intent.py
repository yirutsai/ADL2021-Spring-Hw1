import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

import numpy as np

TRAIN = "train"
DEV = "eval"
TEST="test"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len,split)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN],collate_fn=datasets[TRAIN].collate_fn,batch_size=args.batch_size,shuffle=True)
    valid_dataloader = DataLoader(datasets[DEV],collate_fn=datasets[DEV].collate_fn,batch_size=args.batch_size,shuffle=False)
    print(f"num_class:{len(intent2idx)}")
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings,hidden_size = args.hidden_size,num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional,num_class=len(intent2idx),netType=args.netType,pack=args.pack,attn = args.attn)
    model = model.to(args.device)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    if(args.use_scheduler):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epoch, steps_per_epoch=len(train_dataloader), pct_start=0.1)
    else:
        scheduler = None

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True,exist_ok = True)
    best_acc = 0
    best_round = 0
    for epoch in range(1,args.num_epoch+1):
        # TODO: Training loop - iterate over train dataloader and update model weights
        print(f"Epoch:{epoch:03d}/{args.num_epoch}")
        train_one_epoch(args,model,train_dataloader,optimizer,scheduler)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        valid_acc = valid(args,model,valid_dataloader)
        if(best_acc<valid_acc):
            best_acc = valid_acc
            best_round = epoch
            print(f"Saving model best acc: {best_acc}")
            torch.save(model,ckpt_dir/"model.ckpt")
    print(args)
    print(f"best acc:{best_acc}    best round :{best_round}")
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--grad_clip",type=float,default = 5)
    parser.add_argument("--pack",action="store_true")
    parser.add_argument("--attn",action="store_true")
    parser.add_argument("--netType",type=str,default="LSTM")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float,default=0)
    parser.add_argument("--use_scheduler",action="store_true")

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--seed",type=int,default=777)

    args = parser.parse_args()
    print(args)
    return args

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def accuracy(pred_labels,labels):
    acc = 0
    for i in range(len(pred_labels)):
        if(pred_labels[i]==labels[i]):
            acc+=1
    return acc/len(labels)
def train_one_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    train_losses = []
    pred_labels = []
    labels = []
    for i, batch in enumerate(train_loader):
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)
        labels.extend(batch["intent"].tolist())
        optimizer.zero_grad()
        pred_logits = model(batch)
        pred_labels.extend(pred_logits.max(1,keepdim = True).indices.reshape(-1).tolist())
        # print(pred_logits.shape)
        loss = F.cross_entropy(pred_logits,batch["intent"])
        train_losses.append(loss.item())
        # bar.set_postfix(loss=output_dict['loss'].item(), iter=i, lr=optimizer.param_groups[0]['lr'])

        # am_ce.update(output_dict['loss'], n=batch['intent'].size(0))
        # m.update(batch['intent'].detach().cpu(), output_dict['pred_labels'].detach().cpu())
        loss.backward()

        if(args.grad_clip >=0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()
        if(args.use_scheduler):
            scheduler.step()
    
    avg_train_loss = sum(train_losses)/len(train_losses)
    # scheduler.step()
    acc = accuracy(pred_labels,labels)
    print(f"Train Loss: {avg_train_loss:.4f}\t Acc : {acc:.3f}")
    # print('Train Loss: {:6.4f}\t Aux: {:6.4f}\t Penalization: {:6.4f}\t Acc: {:6.4f}'.format(am_ce.avg, am_aux.avg, am_p.avg, m.acc))
    # return am_ce.avg, am_aux.avg, am_p.avg, m.acc
def valid(args,model,val_loader):
    model.eval()
    pred_labels = []
    labels = []
    for batch in val_loader:
        batch["text"] = batch["text"].to(args.device)
        batch["intent"] = batch["intent"].to(args.device)
        labels.extend(batch["intent"].tolist())
        pred_logits = model(batch)
        pred_labels.extend(pred_logits.max(1,keepdim=True).indices.reshape(-1).tolist())
    acc = accuracy(pred_labels,labels)
    print(f"Valid acc: {acc:.3f}")  
    return acc

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    same_seeds(args.seed)
    main(args)
