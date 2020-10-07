r'''
Both LineByLineTextDataset and TextDataset load a full .txt file into memory.
Not really useful when working with large data in a single file.
Nice solution from BramVanroy https://github.com/huggingface/transformers/issues/3083

added has_empty_lines to deal with the following format

Lorem ipsum ....\n
\n
dolor sit amet\n
\n
sentence3
'''


import linecache
from pathlib import Path

from torch.utils.data import Dataset




class LazyLineByLineTextDataset(Dataset):
    '''Truncates sequences at block_size, does not feed the rest to the model.'''

    def __init__(self, tokenizer, file_path, block_size=512, has_empty_lines=True):
        self.fin = file_path
        self.has_empty_lines = has_empty_lines
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.num_entries = self._get_n_lines(self.fin)


    def _get_n_lines(self, fin):
        with Path(fin).resolve().open(encoding='utf-8') as fhin:
            
            empty_lines = 0
            for line_idx, line in enumerate(fhin, 1):
                if line == '\n':
                    empty_lines+=1
                else:
                    pass

        return (line_idx - empty_lines) if self.has_empty_lines else line_idx

    def __getitem__(self, idx):

        if self.has_empty_lines:
            idx = idx*2

        # linecache starts counting from one, not zero, +1 the given index
        idx += 1
        line = linecache.getline(self.fin, idx)
        line = line.rstrip()
        line = line[:self.block_size] if self.block_size is not None else line
        line = self.tokenizer.encode_plus(line)

        return line

    def __len__(self):
        return self.num_entries


