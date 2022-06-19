from cProfile import label
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from attr import validate

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from dataset import SeqTagDataset
from utils import Vocab
from model import SeqTagger

import numpy as np

TRAIN = "train"
DEV = "eval"
TEST="test"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    idx2tag = {idx:tag for tag,idx in tag2idx.items()}

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTagDataset] = {
        split: SeqTagDataset(split_data, vocab, tag2idx, args.max_len,split)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN],collate_fn=datasets[TRAIN].collate_fn,batch_size=args.batch_size,shuffle=True)
    valid_dataloader = DataLoader(datasets[DEV],collate_fn=datasets[DEV].collate_fn,batch_size=args.batch_size,shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    print(f"num_class:{len(tag2idx)}")
    model = SeqTagger(embeddings=embeddings,hidden_size = args.hidden_size,num_layers=args.num_layers,cnn_num_layers=args.cnn_num_layers,dropout=args.dropout,bidirectional=args.bidirectional,num_class=len(tag2idx),netType=args.netType,pack=args.pack,max_len = args.max_len,use_crf = args.use_crf)
    model = model.to(args.device)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True,exist_ok = True)
    best_acc = 0
    best_round = 0
    criterion=torch.nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(1,args.num_epoch+1):
        # TODO: Training loop - iterate over train dataloader and update model weights
        print(f"Epoch:{epoch:03d}/{args.num_epoch}")
        train_one_epoch(args,model,train_dataloader,optimizer,scheduler=None,criterion=criterion)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        valid_acc = valid(args,model,valid_dataloader)
        if(best_acc<=valid_acc):
            best_acc = valid_acc
            best_round = epoch
            print(f"Saving model best acc: {best_acc}")
            torch.save(model,ckpt_dir/"model.ckpt")
    print(f"best_acc:{best_acc}    best_round:{best_round}")
    print(f"start to report Q4:")
    report(args,valid_dataloader,idx2tag)
def predict(args,model,test_loader):
    model.eval()
    pred_labels = np.empty((0,args.max_len),dtype=np.int)
    labels = np.empty((0,args.max_len),dtype=np.int)
    lens = []
    with torch.no_grad():
        for batch in test_loader:
            batch["tokens"] = batch["tokens"].to(args.device)
            batch["tags"] = batch["tags"].to(args.device)
            batch["mask"] = batch["tokens"].gt(0).float().to(args.device)
            lens.extend(batch["length"])
            labels = np.concatenate([labels,batch["tags"].detach().cpu()])
            output = model(batch)
            pred_labels = np.concatenate([pred_labels,output["pred_labels"]])
    print(f"pred_labels.shape:{pred_labels.shape}")
    print(f"labels.shape:{labels.shape}")
    return pred_labels,labels,lens

def report(args,valid_dataloader,idx2tag):
    model=torch.load(args.ckpt_dir/"model.ckpt").to(args.device)
    model.eval()
    pred_labels,labels,lens = predict(args,model,valid_dataloader)
    y_true = []
    y_pred = []
    for i in range(len(labels)):
        idx = labels[i]<10
        y_true.append(labels[i][idx].tolist())
        y_pred.append(pred_labels[i][idx].tolist())
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            y_true[i][j] = idx2tag[y_true[i][j]]
            y_pred[i][j] = idx2tag[y_pred[i][j]]
    print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cnn_num_layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--grad_clip",type=float,default = 5)
    parser.add_argument("--pack",action="store_true")
    parser.add_argument("--netType",type=str,default="GRU")
    parser.add_argument("--use_crf",action="store_true")

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-3)

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
    joint_acc = 0
    token_acc = 0
    n_token = 0
    for i in range(len(labels)):
        idx = labels[i]<10
        token_cor = np.equal(labels[i][idx],pred_labels[i][idx]).sum()
        n_token += (idx).sum()
        token_acc += token_cor
        if((idx).sum()==token_cor):
            joint_acc+=1
    
    return joint_acc/len(labels),token_acc/n_token

def train_one_epoch(args, model, train_loader, optimizer, scheduler,criterion):
    model.train()
    train_losses = []
    pred_labels = np.empty((0,args.max_len),dtype=np.int)
    labels = np.empty((0,args.max_len),dtype=np.int)
    for i, batch in enumerate(train_loader):
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        batch["mask"] = batch["tokens"].gt(0).float().to(args.device)
        labels = np.concatenate([labels,batch["tags"].detach().cpu()])
        optimizer.zero_grad()
        output = model(batch)
        # pred_logits = output["logits"]
        # print(output["pred_labels"].shape)
        pred_labels = np.concatenate([pred_labels,output["pred_labels"]])
        # print(f"batch['tags']:{batch['tags'].shape}")
        loss = output["loss"]
        train_losses.append(loss.item())
        # bar.set_postfix(loss=output_dict['loss'].item(), iter=i, lr=optimizer.param_groups[0]['lr'])

        # am_ce.update(output_dict['loss'], n=batch['tag'].size(0))
        # m.update(batch['tag'].detach().cpu(), output_dict['pred_labels'].detach().cpu())
        loss.backward()

        if(args.grad_clip >=0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()
        # if args.scheduler_type == "onecycle":
        #     scheduler.step()

    avg_train_loss = sum(train_losses)/len(train_losses)
    joint_acc,token_acc = accuracy(pred_labels,labels)
    print(f"Train Loss: {avg_train_loss:.4f}\t joint_Acc : {joint_acc:.4f}\t token_acc:{token_acc:.4f}")
    # print('Train Loss: {:6.4f}\t Aux: {:6.4f}\t Penalization: {:6.4f}\t Acc: {:6.4f}'.format(am_ce.avg, am_aux.avg, am_p.avg, m.acc))
    # return am_ce.avg, am_aux.avg, am_p.avg, m.acc
def valid(args,model,val_loader):
    model.eval()
    pred_labels = np.empty((0,args.max_len),dtype=np.int)
    labels = np.empty((0,args.max_len),dtype=np.int)
    with torch.no_grad():
        for batch in val_loader:
            batch["tokens"] = batch["tokens"].to(args.device)
            batch["tags"] = batch["tags"].to(args.device)
            batch["mask"] = batch["tokens"].gt(0).float().to(args.device)
            labels = np.concatenate([labels,batch["tags"].detach().cpu()])
            output = model(batch)
            pred_labels = np.concatenate([pred_labels,output["pred_labels"]])
    joint_acc,token_acc = accuracy(pred_labels,labels)
    print(f"Valid joint_acc: {joint_acc:.4f}\t token_acc:{token_acc:.4f}")  
    return joint_acc

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    same_seeds(args.seed)
    main(args) 
    print(args)      