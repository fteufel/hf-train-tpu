'''
pads to max length.
'''

from torch.utils.data import Dataset
import torch
import logging
import os
from pathlib import Path
import linecache
from datasets import load_dataset
logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, 
                                          truncation=True,
                                          padding='max_length', 
                                          max_length = block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LazyLineByLineTextDataset(Dataset):
    '''Truncates sequences at block_size, does not feed the rest to the model.'''

    def __init__(self, tokenizer, file_path, block_size=512):
        self.fin = file_path
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.num_entries = self._get_n_lines(self.fin)


    def _get_n_lines(self, fin):
        with Path(fin).resolve().open(encoding='utf-8') as fhin:
            
            for line_idx, line in enumerate(fhin, 1):
                pass

        return line_idx

    def __getitem__(self, idx):

        if self.has_empty_lines:
            idx = idx*2

        # linecache starts counting from one, not zero, +1 the given index
        idx += 1
        line = linecache.getline(self.fin, idx)
        line = line.rstrip()
        line = self.tokenizer(line, add_special_tokens=True, 
                                          truncation=True,
                                          padding='max_length', 
                                          max_length = block_size)

        #return line
        return torch.tensor(line["input_ids"], dtype=torch.long)


    def __len__(self):
        return self.num_entries


class LazyLineByLineTextHuggingFaceDataset(Dataset):
    '''Truncates sequences at block_size, does not feed the rest to the model.
    Uses datasets library to load data out of core instead of linecache'''

    def __init__(self, tokenizer, file_path, block_size=512):
        ds = load_dataset('text', data_files=[file_path])
        self.ds =  ds['train']
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __getitem__(self, idx):

        # linecache starts counting from one, not zero, +1 the given index
        line = self.ds[idx]
        line = self.tokenizer(line, add_special_tokens=True, 
                                          truncation=True,
                                          padding='max_length', 
                                          max_length = block_size)

        #return line
        return torch.tensor(line["input_ids"], dtype=torch.long)


    def __len__(self):
        return len(self.ds)