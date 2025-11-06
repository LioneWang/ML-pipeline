# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 3:10 下午
# @Author  : jeffery (modified by Gemini)
# @FileName: metric.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr


def tensor_to_numpy(preds, labels):
    """
    (*** 关键修正 ***)
    将模型的 logits (preds) 转换为类别索引 (argmax)，
    然后再将 preds 和 labels 转换为 numpy 数组。
    """
    # preds [batch_size, num_classes] (logits)
    # labels [batch_size] (indices)
    
    # 1. 将 logits 转换为预测的类别索引
    pred_classes = torch.argmax(preds, dim=1)
    
    # 2. 将预测索引和真实标签转换为 numpy
    return pred_classes.cpu().detach().numpy(), labels.cpu().detach().numpy()


def accuracy(preds, labels):
    """
    每个标签都计算accuracy，然后求平均，不考虑数据均衡问题
    (此函数现在可以正常工作，因为它依赖于修正后的 tensor_to_numpy)
    """
    preds, labels = tensor_to_numpy(preds, labels)
    acc = accuracy_score(labels, preds, normalize=True)
    return acc


def macro_precision(preds, labels):
    """
    每个标签都计算precision，然后求平均，不考虑数据均衡问题
    (此函数现在可以正常工作)
    """
    preds, labels = tensor_to_numpy(preds, labels)
    # 添加 zero_division=0 以避免在某些 batch 中某个类没有被预测时发出警告
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    return precision


def macro_recall(preds, labels):
    """
    每个标签都计算recall，然后求平均，不考虑数据均衡问题
    (此函数现在可以正常工作)
    """
    preds, labels = tensor_to_numpy(preds, labels)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    return recall


def macro_f1(preds, labels):
    """
    每个标签都计算f1，然后求平均，不考虑数据均衡问题
    (此函数现在可以正常工作)
    """
    preds, labels = tensor_to_numpy(preds, labels)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return f1


def micro_precision(preds, labels):
    """
    计算全数据的precision
    (此函数现在可以正常工作)
    """
    preds, labels = tensor_to_numpy(preds, labels)
    return precision_score(labels, preds, average='micro', zero_division=0)


def micro_recall(preds, labels):
    """
    计算全数据的recall
    (此函数现在可以正常工作)
    """
    preds, labels = tensor_to_numpy(preds, labels)
    return recall_score(labels, preds, average='micro', zero_division=0)


def micro_f1(preds, labels):
    """
    计算全数据的f1
    (此函数现在可以正常工作)
    """
    preds, labels = tensor_to_numpy(preds, labels)
    return f1_score(labels, preds, average='micro', zero_division=0)



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    (此函数用于二分类，当前项目未使用)
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, y):
    """
    (*** 关键修正 ***)
    使用 PyTorch 计算准确率 (不依赖 sklearn)
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # preds [batch_size, num_classes] (logits)
    # y [batch_size] (indices)
    
    # 1. 使用 argmax 沿着 1 维度找出分数最高的索引（即模型预测的类别）
    #    pred_classes 的形状将是 [batch_size]
    pred_classes = torch.argmax(preds, dim=1) 

    # 2. 比较预测索引和真实索引
    #    .eq() 返回一个布尔张量 (e.g., [True, False, True, ...])
    correct = pred_classes.eq(y)
    
    # 3. 计算百分比
    #    correct.sum() 是正确的数量
    #    y.shape[0] 是 batch size
    #    我们使用 .item() 将其转换为一个 Python float
    return (correct.sum().cpu() / torch.FloatTensor([y.shape[0]])).item()