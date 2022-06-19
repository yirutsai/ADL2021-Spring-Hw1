from cgi import test
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from dataset import SeqTagDataset
from utils import Vocab


def predict(args,model,test_loader):
    model.eval()
    pred_labels = np.empty((0,args.max_len),dtype=np.int)
    lens = []
    with torch.no_grad():
        for batch in test_loader:
            batch["tokens"] = batch["tokens"].to(args.device)
            batch["mask"] = batch["tokens"].gt(0).float().to(args.device)
            lens.extend(batch["length"])
            output = model(batch)
            pred_labels = np.concatenate([pred_labels,output["pred_labels"]])

    return pred_labels,lens
def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTagDataset(data, vocab, tag2idx, args.max_len,"test")
    # TODO: crecate DataLoader for test dataset
    test_dataloader = DataLoader(dataset,collate_fn=dataset.collate_fn,batch_size=args.batch_size,shuffle=False)
    model=torch.load(args.ckpt_path).to(args.device)
    model.eval_mode = True
    model.eval()

    
    # load weights into model

    # TODO: predict dataset
    pred_labels,lens = predict(args,model,test_dataloader)
    output_file(args.pred_file,dataset,pred_labels,lens)
    # TODO: write prediction to file (args.pred_file)

def output_file(fn:str,dataset,pred_labels,lens):
    with open(fn,"w")as f:
        f.write(f"id,tags\n")
        for i in range(len(dataset.data)):
            ans = []
            for j in range(lens[i]):
                ans.append(dataset.idx2label(pred_labels[i][j]))
            f.write(f"{dataset.data[i]['id']},{' '.join(ans)}\n")

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
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

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
