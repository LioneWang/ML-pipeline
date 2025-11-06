# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 2:52 下午
# @Author  : jeffery (modified by Gemini)
# @FileName: __init__.py
# @Description:
import torch
import transformers
from torch.utils.data import DataLoader

# (确保这些 .py 文件存在于 model/ 目录中)
import model.models as module_models
import model.loss as module_loss       
import model.metric as module_metric 

__all__ = ["makeModel", "makeLoss", "makeMetrics", "makeOptimizer","makeLrSchedule"]


def makeModel(config):
    """
    从 config['model_arch'] 创建模型
    """
    return config.init_obj('model_arch', module_models)


def makeLoss(config):
    """
    (*** 修正 1: 移除 '()' ***)
    从 config['loss'] 创建损失函数列表。
    这现在返回一个 *函数列表*，而不是调用它们。
    """
    return [getattr(module_loss, crit) for crit in config['loss']]


def makeMetrics(config):
    """
    (*** 修正 1: 移除 '()' ***)
    从 config['metrics'] 创建指标函数列表。
    """
    return [getattr(module_metric, met) for met in config['metrics']]


def makeOptimizer(config, model):
    """
    (*** 修正 2: 修复 'lr' 歧义 ***)
    创建优化器, 处理不同的 LRs, 并在构造函数前
    从 'args' 字典中 .pop() 'lr'。
    """
    parameters = []
    model_parameters = [*filter(lambda p: p.requires_grad, model.parameters())]
    
    # (1. 复制 args 字典以进行修改)
    optimizer_args = dict(config['optimizer']['args'])
    optimizer_type = config['optimizer']['type']

    # 检查是否为 transformer
    if 'transformer' in config.config['model_arch']['type'].lower():
        # (假设 transformer 模型实例在 model.transformer_model)
        transformers_parameters = [*filter(lambda p: p.requires_grad, model.transformer_model.parameters())]
        model_parameters = list(set(model_parameters)-set(transformers_parameters))
        
        # (2. 从 args 字典中获取并 *移除* 'transformer_lr')
        transformer_lr = optimizer_args.pop('transformer_lr', 2e-5) 
        
        parameters.append({
            'params': transformers_parameters,
            'lr': transformer_lr # 'lr' 是一个 float
        })
    
    # (3. 从 args 字典中获取并 *移除* 'lr')
    # (这是修复 'TypeError: got multiple values' 和
    #  'TypeError: can't multiply sequence' 的关键)
    main_lr = optimizer_args.pop('lr', 1e-3)

    # *** 修复开始：确保 main_lr 是浮点数 ***
    # 处理配置可能将 lr 解析为列表 [1e-3] 或字符串 '1e-3' 的情况
    if isinstance(main_lr, (list, tuple)):
        main_lr = float(main_lr[0])
    else:
        main_lr = float(main_lr)
    # *** 修复结束 ***
    parameters.append({
        'params': model_parameters,
        'lr': main_lr # 'lr' 是一个 float
    })
    
    optimizer_class = getattr(torch.optim, optimizer_type)
    
    # (4. 传递 *已清理* 的 optimizer_args)
    # (这个字典不再包含 'lr', 避免了歧义)
    optimizer = optimizer_class(parameters, **optimizer_args)

    return optimizer


def makeLrSchedule(config, optimizer, train_dataloader: DataLoader):
    """
    (无修改 - 此函数是正确的)
    使用 train_dataloader 和 epochs 正确计算 num_training_steps
    """
    # 1. 从 trainer config 中获取 epochs
    epochs = config['trainer']['epochs']
    
    # 2. 从 dataloader 中获取每个 epoch 的步数
    steps_per_epoch = len(train_dataloader)
    
    # 3. (正确) 计算总训练步数
    num_training_steps = steps_per_epoch * epochs
    
    # 4. 从 scheduler config 中获取 warmup 步数
    num_warmup_steps = config['lr_scheduler']['args']['num_warmup_steps']
    
    # 5. 从 config 中获取 scheduler 类型
    scheduler_type = config['lr_scheduler']['type']
    scheduler_fn = getattr(transformers.optimization, scheduler_type)
    
    print(f"Creating LR Scheduler: Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    return scheduler_fn(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )