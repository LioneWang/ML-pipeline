import argparse
import torch
import numpy as np
from tqdm import tqdm
import collections
import pandas as pd # (用于美化混淆矩阵)
from pathlib import Path
import sys

# (导入我们项目中的模块)
from utils import ConfigParser
import trainer.dataset as module_data_process
from torch.utils.data import DataLoader
import model as module_factory

from torch.nn import functional as F

# (*** 新增导入 ***)
from sklearn.metrics import classification_report, confusion_matrix
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAVE_SEABORN = True
except ImportError:
    print("Warning: Seaborn or Matplotlib not installed. 混淆矩阵图将被禁用。")
    print("Please run: pip install seaborn matplotlib pandas")
    HAVE_SEABORN = False


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # (*** 修改: 移除 logger ***)
    # logger = config.get_logger('inference') 
    
    # --- 1. 准备设备 ---
    device_id = config['main_device_id']
    device = torch.device(f'cuda:{device_id}' if config['num_gpu'] > 0 else 'cpu')
    print(f"Using device: {device}")

    # --- 2. 准备 Dataloader ---
    test_set = config.init_obj('test_set', module_data_process)
    
    test_batch_size = config['test_set']['args']['batch_size']
    num_workers = config['test_set']['args'].get('num_workers', 0)
    
    if not hasattr(test_set, 'collate_fn_4_inference'):
        print(f"ERROR: Dataset '{config['test_set']['type']}' 缺少 'collate_fn_4_inference' 函数。")
        raise AttributeError("collate_fn_4_inference not found in Dataset")

    test_dataloader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        num_workers=num_workers,
        collate_fn=test_set.collate_fn_4_inference, # (使用特殊的 collate_fn)
        shuffle=False
    )
    
    id_map_label = {i: label for i, label in enumerate(test_set.LABEL_LIST)}
    label_names = test_set.LABEL_LIST # (用于报告)
    print(f"Loaded {len(test_set)} test examples.")
    print(f"Label map: {id_map_label}")

    # --- 3. 准备模型 (与 train.py 逻辑一致) ---
    print("Building model...")
    
    # (*** 关键修正 1: 检查是否为 Transformer ***)
    is_transformer = config['model_arch']['type'].lower().startswith('transformers')

    if not is_transformer:
        # (*** 关键修正 2: 仅在非 Transformer 模式下注入嵌入 ***)
        print("非 Transformer 模型, 正在注入 embedding_matrix...")
        if not hasattr(test_set, 'embedding_matrix'):
             raise ValueError("Dataset 缺少 'embedding_matrix' 属性 (用于非 Transformer 模型)。")
        
        embedding_matrix = test_set.embedding_matrix
        vocab_size = len(test_set.word_to_index)
        embedding_dim = embedding_matrix.shape[1]

        # 注入参数到 model config
        config['model_arch']['args']['vocab_size'] = vocab_size
        config['model_arch']['args']['embedding_dim'] = embedding_dim
        config['model_arch']['args']['embedding_matrix'] = embedding_matrix
        if 'word_embedding' in config['model_arch']['args']:
            del config['model_arch']['args']['word_embedding']
    else:
        print("Transformer 模型, 跳过 embedding_matrix 注入。")

    
    model = module_factory.makeModel(config)

    # --- 4. 加载 Checkpoint ---
    print(f"Loading checkpoint: {config.resume} ...")
    checkpoint = torch.load(config.resume, map_location=device, weights_only=False) 
    state_dict = checkpoint['state_dict']
    
    # (更健壮的加载方式)
    if isinstance(model, torch.nn.DataParallel):
         model.module.load_state_dict(state_dict)
    else:
        # 如果 state_dict 键有 'module.' 前缀 (来自多GPU训练)
        if all(key.startswith('module.') for key in state_dict.keys()):
            print("Removing 'module.' prefix from DataParallel checkpoint.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # --- 5. 运行推理 ---
    model = model.to(device)
    model.eval()

    print("Starting inference...")

    # (*** 收集所有预测和标签 ***)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc="Running Inference"):
            
            # (*** 关键修正 3: 解包所有 5 个值 ***)
            input_ids, attention_masks, text_lengths, labels, texts = batch_data

            # 2. 移动到设备
            input_ids = input_ids.to(device)
            text_lengths = text_lengths.to(device)
            labels = labels.to(device)
            # (*** 关键修正 4: 移动 attention_masks ***)
            if attention_masks is not None:
                attention_masks = attention_masks.to(device)
            
            # 3. 模型前向
            # (*** 关键修正 5: 传递 attention_masks ***)
            output, _ = model(input_ids, attention_masks, text_lengths)
            
            # 4. 获取预测
            output = torch.argmax(F.softmax(output, dim=-1), dim=-1)
            
            # 5. 转换回 CPU 以便收集和打印
            preds_np = output.cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()
            
            # 6. (*** 收集结果 ***)
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            
            # 7. (*** 逐个打印 ***)
            for text, pred_id, label_id in zip(texts, preds_np, labels_np):
                tqdm.write("\n" + "--" * 40)
                tqdm.write(f"Text:    {text}")
                tqdm.write(f"Predict: {id_map_label[pred_id]} (id={pred_id})")
                tqdm.write(f"True:    {id_map_label[label_id]} (id={label_id})")

    # --- 6. (*** 打印最终报告 ***) ---
    print("\n\n" + "="*50)
    print(" 最终测试结果 ".center(50, "="))
    print("="*50)

    print("\n" + "="*50)
    print(" Classification Report")
    print("="*50)
    
    report = classification_report(all_labels, all_preds, target_names=label_names, digits=4, zero_division=0)
    print(report)

    print("\n" + "="*50)
    print(" Confusion Matrix")
    print("="*50)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)

def run(config_fname, model_path):
    """
    (与 train.py 的 run 函数逻辑一致)
    """
    args_dict = {
        'config': config_fname,
        'resume': model_path, # (将 model_path 传递给 resume)
        'device': None         # (让 ConfigParser 处理 device_id)
    }
    config = ConfigParser.from_args(args_dict)
    main(config)
if __name__ == '__main__':
    # --- 请修改为您自己的路径 ---
    
    parser = argparse.ArgumentParser(description='Run inference on a trained model.')
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        required=True, 
        help='Path to the configuration file (e.g., configs/rnn.yml)'
    )
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint (e.g., saved/RnnModel/.../model_best.pth)"
    )
    
    args = parser.parse_args()
    
    CONFIG_PATH = args.config
    MODEL_PATH = args.model
    

    run(CONFIG_PATH, MODEL_PATH)
