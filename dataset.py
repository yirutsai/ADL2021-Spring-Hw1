from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab,pad_to_len

TRAIN = "train"
DEV = "eval"
TEST="test"
class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode:str
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # samples = sorted(samples,reverse=True,key = lambda x:len(x["text"]))
        batch:Dict = {}
        batch["id"] = [s["id"] for s in samples]
        batch["text"] = [s["text"].split() for s in samples]
        batch["text"] = torch.tensor(self.vocab.encode_batch(batch["text"],self.max_len))
        batch["length"] = [min(len(s["text"]),self.max_len) for s in samples]
        if(self.mode!=TEST):
            batch["intent"] = [self.label2idx(s["intent"]) for s in samples]
            batch["intent"] = torch.tensor(batch["intent"])
        else:
            batch["intent"] = torch.zeros(len(samples),dtype = torch.int8)
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode:str
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        # self.label_mapping["ignore"]=10
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # samples = sorted(samples,reverse=True,key = lambda x:len(x["text"]))
        batch:Dict = {}
        batch["id"] = [s["id"] for s in samples]
        batch["tokens"] = [s["tokens"] for s in samples]
        batch["tokens"] = torch.tensor(self.vocab.encode_batch(batch["tokens"],self.max_len))
        batch["length"] = [min(len(s["tokens"]),self.max_len) for s in samples]
        if(self.mode!=TEST):
            batch["tags"] = [[self.label2idx(tag) for tag in s["tags"]] for s in samples]
            batch["tags"] = torch.tensor(pad_to_len(batch["tags"],self.max_len,10))
        else:
            batch["tags"] = torch.zeros((len(samples),self.max_len),dtype = torch.int8)
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]