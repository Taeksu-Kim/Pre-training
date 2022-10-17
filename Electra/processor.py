import os
import time
import copy
import math
import random
import pickle
from filelock import FileLock
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset

from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import DataCollatorForLanguageModeling

logger = logging.get_logger(__name__)

class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        num_min_lines: int,
        overwrite_cache=False,
        full_seq_probability=0.5,
    ):
        # caching
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.full_seq_probability = full_seq_probability
        self.num_min_lines = num_min_lines

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer

        lock_path = cached_features_file + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else: # 캐시가 없는 경우
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]] # document 단위로 학습
                with open(file_path, encoding="utf-8") as f:
                    while True: 
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # 이중 개행일 시, documents에 새로 document 추가
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])

                        # line 별로 document에 추가
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []

                for doc_index, document in tqdm(enumerate(self.documents)):
                    self.create_examples_from_document(document, doc_index) 

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int):
        """Creates examples for a single document."""
        
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False)

        current_chunk = []  # a buffer stored current working segments
        add_chunk = []
        last_length = 0
        i = 0

        while i < len(document):
            line = document[i]
            current_length = len(line)
            if current_length > max_num_tokens:
                line = line[:max_num_tokens]

            if last_length + current_length >= max_num_tokens:
                add_chunk = current_chunk
                current_chunk = []
                last_length = 0

            current_chunk.append(line)
            last_length += current_length

            if i == len(document) - 1:
                add_chunk = current_chunk

            prep_texts = []

            if len(add_chunk) != 0:

                if random.random() <= self.full_seq_probability:
                    prep_texts.append([])
                  
                    for line in add_chunk:
                        prep_texts[-1].extend(line)
                    
                else:

                    init_length = len(add_chunk)
                    cur_length = copy.deepcopy(init_length)

                    cut_lens = []

                    while cur_length > 0:
                        cut_length = max(self.num_min_lines, random.randint(max(1, math.floor(cur_length/random.randint(1,5))), cur_length))
                        if cut_length == init_length and init_length > self.num_min_lines:
                            continue
                        if cut_length > cur_length:
                            cut_length = cur_length
                        cur_length -= cut_length
                        cut_lens.append(cut_length)

                        start_idx = 0
                        prep_texts = []

                        for cut_len in cut_lens:

                            end_idx = start_idx + cut_len
                            
                            merge_line = []
                            for line in add_chunk[start_idx:end_idx]:
                              merge_line.extend(line)

                            prep_texts.append(merge_line)
                            
                            start_idx = end_idx 

                for prep_text in prep_texts:
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(prep_text)
 
                    attention_mask = ([1] * len(input_ids)) + ([0] * (self.block_size - len(input_ids)))
                    token_type_ids = [0] * (self.block_size)
                    
                    input_ids += [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
                    
                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    }

                    self.examples.append(example)

                add_chunk = []

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# if __name__ == "__main__":
#   from transformers import ElectraModel, ElectraConfig, ElectraForMaskedLM, ElectraTokenizerFast
#   tokenizer_save_dir = './Tokenizer'
#   model_max_input_len = 512
  
#   tokenizer = ElectraTokenizerFast.from_pretrained(tokenizer_save_dir)

#   dataset = TextDatasetForNextSentencePrediction(
#       tokenizer=tokenizer,
#       file_path='/content/Pre-training/Electra/my_data/wiki_20190620_small.txt',
#       block_size=model_max_input_len,
#       num_min_lines=5
#       overwrite_cache=False,
#       full_seq_probability=0.5,
#   )