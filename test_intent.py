from cgi import test
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

def predict(args,model,test_loader):
    model.eval()
    pred_labels = []
    for batch in test_loader:
        batch["text"] = batch["text"].to(args.device)
        batch["intent"] = batch["intent"].to(args.device)
        pred_logits= model(batch)
        pred_labels.extend(pred_logits.max(1,keepdim = True).indices.reshape(-1).tolist())
    return pred_labels
def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len,"test")
    # TODO: crecate DataLoader for test dataset
    test_dataloader = DataLoader(dataset,collate_fn=dataset.collate_fn,batch_size=args.batch_size,shuffle=False)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # model = SeqClassifier(embeddings=embeddings,hidden_size = args.hidden_size,num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional,num_class=len(intent2idx),netType="LSTM")
    # model = model.to(args.device)
    model=torch.load(args.ckpt_path).to(args.device)
    model.eval()

    
    # load weights into model

    # TODO: predict dataset
    pred_labels = predict(args,model,test_dataloader)
    output_file(args.pred_file,dataset,pred_labels)
    # TODO: write prediction to file (args.pred_file)

def output_file(fn:str,dataset,pred_labels):
    with open(fn,"w")as f:
        f.write(f"id,intent\n")
        for i in range(len(dataset.data)):
            f.write(f"{dataset.data[i]['id']},{dataset.idx2label(pred_labels[i])}\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
