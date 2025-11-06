# -*- coding: utf-8 -*-
# @Author  : jeffery (modified by Gemini)
# @FileName: train.py

import torch
import numpy as np
import random

import argparse
# (导入我们自己的模块)
from utils import ConfigParser # 假设您有
import trainer.dataset as module_data_process 
from trainer.classifier_trainer import ClassifierTrainer 
import model as module_factory

from torch.utils.data import DataLoader

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def makeDataLoader(config):
    """
    动态地从 config 中创建 *训练* 和 *验证* dataloader
    (*** 已移除 Test Dataloader ***)
    """
    # 1. 从 config 获取类名
    train_dataset_class_name = config['train_set']['type']
    valid_dataset_class_name = config['valid_set']['type']
    
    # 2. 从 module_data_process (trainer.dataset) 模块中获取类
    train_dataset_class = getattr(module_data_process, train_dataset_class_name)
    valid_dataset_class = getattr(module_data_process, valid_dataset_class_name)

    # 3. 初始化 Dataset 实例
    train_set = train_dataset_class(**config['train_set']['args'])
    valid_set = valid_dataset_class(**config['valid_set']['args'])
    
    # (*** 已移除 Test Set ***)
    print(f"Train num: {len(train_set)}\t Valid num: {len(valid_set)}")

    # 4. 从 config 中获取参数
    train_batch_size = config['train_set']['args']['batch_size']
    valid_batch_size = config['valid_set']['args']['batch_size']
    num_workers = config['train_set']['args'].get('num_workers', 0) 

    # 5. 创建 DataLoaders
    train_dataloader = DataLoader(
        train_set, 
        batch_size=train_batch_size,
        num_workers=num_workers, 
        collate_fn=train_set.collate_fn,
        shuffle=config['train_set']['args'].get('shuffle', True)
    )
    valid_dataloader = DataLoader(
        valid_set, 
        batch_size=valid_batch_size,
        num_workers=num_workers, 
        collate_fn=valid_set.collate_fn,
        shuffle=config['valid_set']['args'].get('shuffle', False)
    )
    
    # (*** 已移除 Test Dataloader ***)

    # (返回 train_set 以便 main 函数提取 embedding_matrix)
    return train_dataloader, valid_dataloader, train_set


def main(config):
    logger = config.get_logger('train')

    # --- 1. 建立 DataLoaders ---
    # (*** 修改: 不再接收 test_dataloader ***)
    train_dataloader, valid_dataloader, train_dataset = makeDataLoader(config)

    # --- 2. 建立模型 (Model) ---
    is_transformer = config['model_arch']['type'].lower().startswith('transformers')
    
    if not is_transformer:
        # (注入 embedding_matrix 的逻辑是必需的)
        if not hasattr(train_dataset, 'embedding_matrix'):
            logger.error("Dataset 缺少 'embedding_matrix' 属性。")
            raise ValueError("非 Transformer 模型需要 Dataset 提供 embedding_matrix。")
        
        embedding_matrix = train_dataset.embedding_matrix
        vocab_size = len(train_dataset.word_to_index)
        embedding_dim = embedding_matrix.shape[1]

        # 注入参数到 model config
        config['model_arch']['args']['vocab_size'] = vocab_size
        config['model_arch']['args']['embedding_dim'] = embedding_dim
        config['model_arch']['args']['embedding_matrix'] = embedding_matrix
        
        if 'word_embedding' in config['model_arch']['args']:
            del config['model_arch']['args']['word_embedding']
            
        logger.info(f"注入的嵌入参数: vocab_size={vocab_size}, embedding_dim={embedding_dim}")

    
    # (使用 factory)
    model = module_factory.makeModel(config)
    logger.info(model)
    
    # --- 3. 建立其他组件 (使用 factory) ---
    criterion = module_factory.makeLoss(config)
    metrics = module_factory.makeMetrics(config)
    optimizer = module_factory.makeOptimizer(config, model)

    # --- 4. 建立 LR Scheduler (使用 factory) ---
    lr_scheduler = module_factory.makeLrSchedule(config, optimizer, train_dataloader)
                
    # --- 5. 实例化 Trainer ---
    trainer = ClassifierTrainer(model, criterion[0], metrics, optimizer, # (注意: 传入 criterion[0])
                                config=config,
                                data_loader=train_dataloader,
                                valid_data_loader=valid_dataloader,
                                # (*** 关键修正: 传入 None ***)
                                test_data_loader=None, 
                                lr_scheduler=lr_scheduler)
    
    # --- 6. 训练 ---
    trainer.train()
    
    # (*** 新增: 告知用户下一步 ***)
    logger.info("="*50)
    logger.info("训练完成。")
    logger.info(f"最佳模型已保存在: {config.save_dir}/model_best.pth")
    logger.info(f"要运行最终测试, 请运行 python inference.py -c {config_fname} -m {config.save_dir}/model_best.pth")
    logger.info("="*50)


def run(config_fname):
    """
    (与您之前的版本相同)
    """
    args_dict = {
        'config': config_fname,
        'resume': None, 
        'device': '0'
    }
    config = ConfigParser.from_args(args_dict)
    main(config)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a trained model.')
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        required=True, 
        help='Path to the configuration file (e.g., configs/rnn.yml)'
    )
    args= parser.parse_args()
    config_fname=args.config
    run(config_fname)
    #run('configs/rnn.yml')
# python train.py -c configs/transformers.yml