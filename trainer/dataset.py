# -*- coding: utf-8 -*-
# @Author  : Gemini (adapted from jeffery's cnews example)
# @Description: Dataloader for dair-ai 6-class emotion dataset

import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import nltk
from transformers import AutoTokenizer

# --- 1. Dataclass Definitions (from template) ---

@dataclass
class InputExample:
    """
    一个单独的训练/测试样本，从 .jsonl 文件中读取。
    """
    guid: Optional[str]
    text: str
    label: int


@dataclass
class InputFeatures:
    """
    一个被处理过的、可用于模型输入的样本。
    (与 cnews 模板一致)
    """
    guid: Optional[str]
    input_ids: List[int]
    attention_mask: Optional[List[int]]
    label: int
    text: str # (*** 新增 ***: 为推理添加原始文本)

    def __post_init__(self):
        self.sent_len = len(self.input_ids)

# --- 2. 适用于静态嵌入的模型 (RNN, CNN) ---

class DairAiEmbeddingDataset(Dataset):
    """
    该类用于非transformers模型, 词嵌入使用我们从 preprocess.py 生成的 .pkl 文件。
    (结构模仿 CnewsEmbeddingDataset)
    """
    
    LABEL_LIST = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def __init__(self, data_dir: str, file_name: str, cache_dir: str, 
                 word_embedding_path: str, max_seq_len: int = 256, 
                 overwrite_cache: bool = False, **kwargs): 
        
        self.data_dir = Path(data_dir)
        self.file_name = file_name
        self.max_seq_len = max_seq_len
        
        self.label_map_id = {label: i for i, label in enumerate(self.LABEL_LIST)}
        self.num_labels = len(self.LABEL_LIST)

        if cache_dir is None:
            self.cache_dir = self.data_dir / ".cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cache_file = self.cache_dir / (file_name.split('.')[0] + '_static.cache')
        
        print(f"Loading word embedding lookup from {word_embedding_path}")
        glove_lookup = pickle.load(Path(word_embedding_path).open('rb'))
        
        self.word_to_index, self.embedding_matrix = self._build_embedding_matrix(glove_lookup)
        
        if self.feature_cache_file.exists() and not overwrite_cache:
            print(f"Loading features from cached file {self.feature_cache_file}")
            self.features = self._load_features_from_cache()
        else:
            print("Creating features from file...")
            self._check_nltk_punkt()
            self.features = self.convert_examples_to_features(self.read_examples_from_file())
            if not self.features:
                 print("WARNING: No features were created. Check 'read_examples_from_file' logic.")
            self._save_features_to_cache()
            
    def _build_embedding_matrix(self, glove_lookup: Dict[str, np.ndarray]):
        word_to_index = {'<pad>': 0, '<unk>': 1}
        dim = len(glove_lookup['<pad>'])
        
        embedding_matrix = [
            np.zeros(dim, dtype=np.float32),
            np.random.randn(dim).astype(np.float32)
        ]
        
        i = 2
        for word, vector in glove_lookup.items():
            if word == '<pad>':
                continue
            word_to_index[word] = i
            embedding_matrix.append(vector)
            i += 1
            
        embedding_matrix_tensor = torch.tensor(np.array(embedding_matrix), dtype=torch.float32)
        print(f"Built embedding matrix of shape: {embedding_matrix_tensor.shape}")
        return word_to_index, embedding_matrix_tensor

    def _check_nltk_punkt(self):
        try:
            nltk.word_tokenize("test")
        except LookupError:
            print("NLTK 'punkt' tokenizer not found. Downloading...")
            nltk.download('punkt', quiet=True)

    def read_examples_from_file(self) -> List[InputExample]:
        """
        (修正版: 查找 'label' (单数) 并将其视为 int)
        """
        input_file = self.data_dir / self.file_name
        examples = []
        with input_file.open('r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading examples"):
                json_line = json.loads(line)
                
                guid = json_line.get('id')
                text = json_line.get('text')
                label_id = json_line.get('label') # 查找 'label' (单数)

                if text is None or label_id is None:
                    continue
                
                label_id = int(label_id)

                if 0 <= label_id < self.num_labels:
                    examples.append(InputExample(guid=guid, text=text, label=label_id))
                else:
                    continue
        
        print(f"Read {len(examples)} examples from {self.file_name}")
        return examples

    def convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        features = []
        unk_token_id = self.word_to_index['<unk>']
        
        for example in tqdm(examples, desc="Converting examples to features"):
            tokens = nltk.word_tokenize(example.text.lower())
            
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]

            input_ids = [self.word_to_index.get(token, unk_token_id) for token in tokens]
            
            label = example.label
            
            # (*** 修改 ***: 传入 example.text)
            features.append(InputFeatures(guid=example.guid, 
                                          input_ids=input_ids, 
                                          attention_mask=None, 
                                          label=label, 
                                          text=example.text)) # <-- 新增
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def _save_features_to_cache(self):
        with self.feature_cache_file.open('wb') as f:
            pickle.dump(self.features, f)
        print(f"Features saved to {self.feature_cache_file}")

    def _load_features_from_cache(self):
        with self.feature_cache_file.open('rb') as f:
            return pickle.load(f)

    def collate_fn(self, datas: List[InputFeatures]):
        """
        (这是用于 *训练* 的 collate_fn)
        """
        max_len = max([data.sent_len for data in datas])
        pad_token_id = self.word_to_index['<pad>']

        input_ids_list = []
        text_lengths_list = []
        labels_list = []

        for data in datas:
            input_ids = data.input_ids
            pad_len = max_len - data.sent_len
            
            input_ids_list.append(input_ids + [pad_token_id] * pad_len)
            text_lengths_list.append(data.sent_len)
            labels_list.append(data.label)

        input_ids = torch.LongTensor(np.asarray(input_ids_list))
        text_lengths = torch.LongTensor(np.asarray(text_lengths_list))
        labels = torch.LongTensor(np.asarray(labels_list))
        
        return input_ids, None, text_lengths, labels
    
    def collate_fn_4_inference(self, datas: List[InputFeatures]):
        """
        (*** 新增 ***: 这是用于 *推理* 的 collate_fn)
        """
        max_len = max([data.sent_len for data in datas])
        pad_token_id = self.word_to_index['<pad>']

        input_ids_list = []
        text_lengths_list = []
        labels_list = []
        texts_list = [] # <-- 新增

        for data in datas:
            input_ids = data.input_ids
            pad_len = max_len - data.sent_len
            
            input_ids_list.append(input_ids + [pad_token_id] * pad_len)
            text_lengths_list.append(data.sent_len)
            labels_list.append(data.label)
            texts_list.append(data.text) # <-- 新增

        input_ids = torch.LongTensor(np.asarray(input_ids_list))
        text_lengths = torch.LongTensor(np.asarray(text_lengths_list))
        labels = torch.LongTensor(np.asarray(labels_list))
        
        # (返回 5 个值，包括 texts_list)
        return input_ids, None, text_lengths, labels, texts_list


# --- 3. 适用于 Transformers 模型的 (BERT, RoBERTa) ---

class DairAiTransformerDataset(Dataset):
    """
    该类用于 transformers 模型。
    """
    
    LABEL_LIST = DairAiEmbeddingDataset.LABEL_LIST

    def __init__(self, data_dir: str, file_name: str, cache_dir: str, 
                 transformer_model_name: str, max_seq_len: int = 256, 
                 overwrite_cache: bool = False, **kwargs):
        
        self.data_dir = Path(data_dir)
        self.file_name = file_name
        self.max_seq_len = max_seq_len
        self.num_labels = len(self.LABEL_LIST)
        
        print(f"Loading tokenizer for {transformer_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        if cache_dir is None:
            self.cache_dir = self.data_dir / ".cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cache_file = self.cache_dir / (file_name.split('.')[0] + f'_{transformer_model_name.split("/")[-1]}.cache')

        if self.feature_cache_file.exists() and not overwrite_cache:
            print(f"Loading features from cached file {self.feature_cache_file}")
            self.features = self._load_features_from_cache()
        else:
            print("Creating features from file...")
            self.features = self.convert_examples_to_features(self.read_examples_from_file())
            self._save_features_to_cache()
            
    def read_examples_from_file(self) -> List[InputExample]:
        """
        (修正版: 查找 'label' (单数) 并将其视为 int)
        """
        input_file = self.data_dir / self.file_name
        examples = []
        with input_file.open('r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading examples"):
                json_line = json.loads(line)
                
                guid = json_line.get('id')
                text = json_line.get('text')
                label_id = json_line.get('label')

                if text is None or label_id is None:
                    continue
                
                label_id = int(label_id)

                if 0 <= label_id < self.num_labels:
                    examples.append(InputExample(guid=guid, text=text, label=label_id))
                else:
                    continue
        
        print(f"Read {len(examples)} examples from {self.file_name}")
        return examples

    def convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        features = []
        
        for example in tqdm(examples, desc="Converting examples to features"):
            
            inputs = self.tokenizer(
                example.text, 
                max_length=self.max_seq_len,
                padding=False,
                truncation=True,
                return_token_type_ids=False,
                return_attention_mask=True
            )
            
            # (*** 修改 ***: 传入 example.text)
            features.append(InputFeatures(
                guid=example.guid, 
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                label=example.label,
                text=example.text # <-- 新增
            ))
        return features
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def _save_features_to_cache(self):
        with self.feature_cache_file.open('wb') as f:
            pickle.dump(self.features, f)
        print(f"Features saved to {self.feature_cache_file}")

    def _load_features_from_cache(self):
        with self.feature_cache_file.open('rb') as f:
            return pickle.load(f)

    def collate_fn(self, datas: List[InputFeatures]):
        """
        (这是用于 *训练* 的 collate_fn)
        """
        max_len = max([data.sent_len for data in datas])
        pad_token_id = self.tokenizer.pad_token_id

        input_ids_list = []
        attention_masks_list = []
        text_lengths_list = []
        labels_list = []

        for data in datas:
            input_ids = data.input_ids
            attention_mask = data.attention_mask
            pad_len = max_len - data.sent_len
            
            input_ids_list.append(input_ids + [pad_token_id] * pad_len)
            attention_masks_list.append(attention_mask + [0] * pad_len)
            text_lengths_list.append(data.sent_len)
            labels_list.append(data.label)

        input_ids = torch.LongTensor(np.asarray(input_ids_list))
        attention_masks = torch.LongTensor(np.asarray(attention_masks_list))
        text_lengths = torch.LongTensor(np.asarray(text_lengths_list))
        labels = torch.LongTensor(np.asarray(labels_list))
        
        return input_ids, attention_masks, text_lengths, labels

    def collate_fn_4_inference(self, datas: List[InputFeatures]):
        """
        (*** 新增 ***: 这是用于 *推理* 的 collate_fn)
        """
        max_len = max([data.sent_len for data in datas])
        pad_token_id = self.tokenizer.pad_token_id

        input_ids_list = []
        attention_masks_list = []
        text_lengths_list = []
        labels_list = []
        texts_list = [] # <-- 新增

        for data in datas:
            input_ids = data.input_ids
            attention_mask = data.attention_mask
            pad_len = max_len - data.sent_len
            
            input_ids_list.append(input_ids + [pad_token_id] * pad_len)
            attention_masks_list.append(attention_mask + [0] * pad_len)
            text_lengths_list.append(data.sent_len)
            labels_list.append(data.label)
            texts_list.append(data.text) # <-- 新S

        input_ids = torch.LongTensor(np.asarray(input_ids_list))
        attention_masks = torch.LongTensor(np.asarray(attention_masks_list))
        text_lengths = torch.LongTensor(np.asarray(text_lengths_list))
        labels = torch.LongTensor(np.asarray(labels_list))
        
        # (返回 5 个值，包括 texts_list)
        return input_ids, attention_masks, text_lengths, labels, texts_list