import os
import pandas as pd
import argparse
import json
import gzip
import pickle
import numpy as np
import nltk
import re  # 导入正则表达式库

# List of tasks to process for parquet conversion
TASKS = ['train', 'test', 'validation']

def convert_parquet_to_jsonl(base_data_dir):
    """
    Converts parquet files to JSONL format.
    """
    
    # 根据传入的参数定义路径
    DATA_DIR = base_data_dir
    SPLIT_DIR = os.path.join(DATA_DIR, 'split')
    OUTPUT_DIR = DATA_DIR # Save to the parent 'dair-ai' folder

    print(f"Starting data conversion...")
    print(f"Base data directory: {DATA_DIR}")
    print(f"Reading from: {SPLIT_DIR}")
    print(f"Writing to: {OUTPUT_DIR}\n")
    
    # Ensure the output directory exists (it should, but good practice)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # (此函数仍然使用 TASKS 来转换所有 parquet 文件)
    for task in TASKS:
        # Construct file paths
        parquet_file = f'{task}-00000-of-00001.parquet'
        input_path = os.path.join(SPLIT_DIR, parquet_file)
        
        jsonl_file = f'{task}.jsonl'
        output_path = os.path.join(OUTPUT_DIR, jsonl_file)

        # Check if the source file exists
        if not os.path.exists(input_path):
            print(f"Warning: File not found at {input_path}. Skipping '{task}'.")
            continue

        print(f"Processing '{task}' data...")
        
        # Read the Parquet file
        df = pd.read_parquet(input_path)
        
        # Write to JSONL format
        # orient='records' creates a list of dicts
        # lines=True writes each dict as a new line
        df.to_json(output_path, orient='records', lines=True)
        
        print(f"Successfully converted and saved to {output_path}")

    print("\nData conversion finished.")


def build_vocab_from_jsonl(base_data_dir, tasks):
    """
    Reads JSONL files (specified by tasks) and builds a set of all unique words and characters.
    Assumes the text field is named 'text'.
    """
    word_vocab = set()
    char_vocab = set()

    # Ensure NLTK 'punkt' tokenizer is available (will crash if not found)
    # (Removed try...except LookupError)
    # If 'punkt' is not downloaded, run this once manually in Python:
    # import nltk; nltk.download('punkt')
    nltk.word_tokenize("test") 


    print(f"Building vocabulary from text field: 'text'")
    
    for task in tasks:
        jsonl_path = os.path.join(base_data_dir, f'{task}.jsonl')
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found. Skipping for vocab.")
            continue
        
        print(f"Reading {jsonl_path} for vocab...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                # (Removed try...except JSONDecodeError)
                data = json.loads(line)
                    
                # (修改点：硬编码 'text')
                if 'text' not in data or data['text'] is None:
                    # print(f"Warning: 'text' not in line or is null. Skipping.")
                    continue
                
                # (修改点：硬编码 'text')
                text = data['text'].lower()
                
                # Tokenize for words
                words = nltk.word_tokenize(text)
                word_vocab.update(words)
                
                # Get characters
                chars = list(text)
                char_vocab.update(chars)
                
    return word_vocab, char_vocab

def filter_and_save_embeddings(base_data_dir, glove_file_path, word_vocab, char_vocab, dim):
    """
    (This function was previously 'process_glove_embeddings')
    Filters GloVe embeddings based on the provided vocabularies and saves them
    as .pkl files, including <pad> and <unk> tokens.
    """
    output_dir = os.path.join(base_data_dir, 'embedding')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize embedding dictionaries with <pad> and <unk>
    # (修改点：移除了 '<unk>' 标记)
    word_embeddings = {
        '<pad>': np.zeros(dim, dtype=np.float32),
        # '<unk>': np.random.randn(dim).astype(np.float32)
    }
    # (修改点：移除了 '<unk>' 标记)
    char_embeddings = {
        '<pad>': np.zeros(dim, dtype=np.float32),
        # '<unk>': np.random.randn(dim).astype(np.float32)
    }
    
    found_words = 0
    found_chars = 0
    
    print(f"Opening GloVe file: {glove_file_path} (This may take a while...)")
    
    # (Removed try...except FileNotFoundError / Exception)
    with gzip.open(glove_file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            split_line = line.strip().split()
            if len(split_line) < dim + 1: # 确保行长度足够
                print(f"Warning: Skipping malformed line with {len(split_line)} parts.")
                continue
            
            word = split_line[0]
            
            # Check for word vocab
            if word in word_vocab and word not in word_embeddings:
                # (Removed try...except ValueError)
                vector = np.array(split_line[1:], dtype=np.float32)
                if vector.shape[0] == dim:
                    word_embeddings[word] = vector
                    found_words += 1
                else:
                    print(f"Warning: Skipping word '{word}' with wrong dimension {vector.shape[0]} (expected {dim})")
                        
            # Check for char vocab
            if word in char_vocab and word not in char_embeddings:
                # (Removed try...except ValueError)
                vector = np.array(split_line[1:], dtype=np.float32)
                if vector.shape[0] == dim:
                    char_embeddings[word] = vector
                    found_chars += 1


    print("Finished processing GloVe file.")
    print("--- Word Embedding Stats ---")
    print(f"Total unique words in dataset: {len(word_vocab)}")
    print(f"Found {found_words} matching word embeddings in GloVe. (这是您要求的'匹配dair-ai的data的所有word的数目')")
    print(f"OOV (Out of Vocabulary) words: {len(word_vocab) - found_words}")

    print("\n--- Char Embedding Stats ---")
    print(f"Total unique chars in dataset: {len(char_vocab)}")
    print(f"Found {found_chars} matching char embeddings in GloVe.")
    print(f"OOV (Out of Vocabulary) chars: {len(char_vocab) - found_chars}")
    print("(Note: OOV chars are common, as GloVe doesn't contain all symbols, e.g., '!')")

    # Save to PKL
    word_pkl_path = os.path.join(output_dir, 'word_embedding.pkl')
    char_pkl_path = os.path.join(output_dir, 'char_embedding.pkl')
    
    # (Removed try...except Exception)
    with open(word_pkl_path, 'wb') as f:
        pickle.dump(word_embeddings, f)
    print(f"\nWord embeddings saved to: {word_pkl_path}")

    with open(char_pkl_path, 'wb') as f:
        pickle.dump(char_embeddings, f)
    print(f"Char embeddings saved to: {char_pkl_path}")


# (新函数：封装了所有 embedding 逻辑)
def process_embeddings(data_dir, embedding_path):
    """
    Main function to process embeddings.
    Infers dimension, builds vocab, filters, and saves embeddings.
    """
    print(f"\nStarting embedding processing (path provided: {embedding_path})...")
        
    # --- 1. 从 embedding_path 推断嵌入维度 ---
    embedding_dim = None
    match = re.search(r'(\d+)d', os.path.basename(embedding_path))
    
    if match:
        embedding_dim = int(match.group(1))
        print(f"Inferred embedding dimension: {embedding_dim} from filename. (这是您要求的'embedding的维度')")
    else:
        print(f"Error: Could not infer embedding dimension from --embedding_path: {embedding_path}")
        print("Path filename must contain '...[number]d...' (e.g., 'glove.6B.100d.txt.gz').")
        return # 停止

    # --- 2. 从 JSONL 构建词汇表 (使用 'train', 'test', 'validation') ---
    print("Step 1: Building vocabulary from train, test, and validation jsonl files...")
    # (修改点：移除 text_field 参数)
    word_vocab, char_vocab = build_vocab_from_jsonl(
        data_dir, TASKS
    )
    if not word_vocab and not char_vocab:
        print("Error: Vocabulary is empty. Cannot process embeddings.")
        print(f"Please check if '{data_dir}' contains valid .jsonl files")
        print(f"and if 'text' is the correct text field.") # (保留此消息以提示用户)
        return # 停止
    else:
        print(f"Found {len(word_vocab)} unique words and {len(char_vocab)} unique chars from {TASKS}.")
        
    # --- 3. 筛选 GloVe 嵌入 ---
    print("\nStep 2: Filtering GloVe embeddings...")
    # (调用重命名的函数)
    filter_and_save_embeddings(
        data_dir, 
        embedding_path, 
        word_vocab, 
        char_vocab, 
        embedding_dim
    )
    print("\nEmbedding processing finished.")


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess data for the ML pipeline.")
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='data/dair-ai', 
        help="The base directory for the data (default: 'data/dair-ai')"
    )
    
    parser.add_argument(
        '--source_format', 
        type=str, 
        default=None, # 改为非必需
        help="The source file format. Use 'parquet' to convert from parquet to jsonl."
    )
    
    parser.add_argument(
        '--embedding_path', 
        type=str, 
        default=None, # 默认值为 None
        help="Path to the gzipped GloVe file (relative to project root). "
             "If provided, embedding processing will be triggered."
    )
    
    # (修改点：移除了 --text_field 参数)

    args = parser.parse_args()
    return parser, args


def main(parser, args):
    """
    Main execution logic driven by parsed arguments.
    """
    # 1. 检查是否需要转换 Parquet
    if args.source_format and args.source_format.lower() == 'parquet':
        convert_parquet_to_jsonl(args.data_dir)
    
    # 2. 检查是否需要处理嵌入 (仅当 embedding_path 被指定时)
    if args.embedding_path: 
        # (修改点：移除了 text_field 参数)
        process_embeddings(args.data_dir, args.embedding_path)

    # 3. 如果两个操作都没指定
    if (not args.source_format or args.source_format.lower() != 'parquet') and not args.embedding_path:
        print("No action specified.")
        print("Use --source_format parquet to convert data.")
        print("Use --embedding_path /path/to/glove.txt.gz to process embeddings.")
        parser.print_help()


if __name__ == "__main__":
    parser, args = parse_args()
    main(parser, args)