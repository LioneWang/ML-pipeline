import torch
import numpy as np
from tqdm import tqdm # 推荐使用 tqdm 来展示进度
from base import BaseTrainer
from utils import MetricTracker, inf_loop # 假设这些在您的 utils 中

class ClassifierTrainer(BaseTrainer):
    """
    专用于分类任务的 Trainer。
    (从 'Trainer' 重命名为 'ClassifierTrainer' 以提高可读性)
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None):
        
        # 调用 BaseTrainer 的 __init__
        # 假设 BaseTrainer 保存了 model, criterion, metric_ftns, optimizer, config, writer, device
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        
        self.config = config # config 可能会在 BaseTrainer 中保存，但这里为了明确
        self.data_loader = data_loader
        
        if len_epoch is None:
            # 基于 epoch 的训练
            self.len_epoch = len(self.data_loader)
        else:
            # 基于迭代次数的训练
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
            
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_inference = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        
        # (建议修改) self.log_step 可以设置得更通用
        # self.log_step = int(np.sqrt(data_loader.batch_size))
        # 推荐使用 10% 或 20% 的 epoch 长度
        self.log_step = int(self.len_epoch * 0.1) if self.len_epoch > 10 else 1

        # MetricTrackers
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        (已补全)
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(self.data_loader, desc=f'Epoch {epoch} [Train]', total=self.len_epoch, leave=False)
        
        for batch_idx, data in enumerate(progress_bar):
            if batch_idx >= self.len_epoch: # 适用于 iteration-based training
                break

            # 1. 将数据移动到设备
            # (这是更健壮的写法, 假设 BaseTrainer 定义了 self.device)
            input_ids, attention_masks, text_lengths, labels = self._move_batch_to_device(data)

            # 2. 训练
            self.optimizer.zero_grad()
            preds, embedding = self.model(input_ids, attention_masks, text_lengths)
            
            # (修复: 移除 .squeeze() 以避免 batch_size=1 时的 bug)
            # preds = preds.squeeze() 
            
            # (修复: 假设 self.criterion 是函数，而不是列表)
            loss = self.criterion(preds, labels)

            loss.backward()
            self.optimizer.step()

            # 3. 更新指标和日志
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                # 传递 logits (preds) 和 labels
                self.train_metrics.update(met.__name__, met(preds, labels))

            # 4. 更新进度条显示
            if batch_idx % self.log_step == 0 or batch_idx == self.len_epoch - 1:
                progress_bar.set_postfix(**self.train_metrics.result())

        # (循环后)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            
        if self.do_inference:
            test_log = self._inference_epoch(epoch)
            log.update(**{'test_' + k: v for k, v in test_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        progress_bar = tqdm(self.valid_data_loader, desc=f'Epoch {epoch} [Valid]', leave=False)

        with torch.no_grad():
            for batch_idx, data in enumerate(progress_bar):
                input_ids, attention_masks, text_lengths, labels = self._move_batch_to_device(data)

                preds, embedding = self.model(input_ids, attention_masks, text_lengths)
                
                # (修复: 移除 .squeeze())
                # preds = preds.squeeze()
                
                # (修复: 假设 self.criterion 是函数)
                loss = self.criterion(preds, labels)

                # update metric
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    # 传递 logits (preds) 和 labels
                    self.valid_metrics.update(met.__name__, met(preds, labels))

        # (注意: 在 BaseTrainer 中添加 histogram 逻辑可能更通用)
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _inference_epoch(self, epoch):
        """
        (已修复)
        Inference after training an epoch
        """
        self.model.eval()
        self.test_metrics.reset()
        
        progress_bar = tqdm(self.test_data_loader, desc=f'Epoch {epoch} [Test]', leave=False)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(progress_bar):
                input_ids, attention_masks, text_lengths, labels = self._move_batch_to_device(data)

                # (关键修复: 这里的逻辑与 _valid_epoch 完全一致)
                # 无论是 transformer 还是非 transformer, 
                # attention_masks 要么是 Tensor, 要么是 None, 模型应该自己处理。
                # 之前 'else: attention_masks = text_lengths' 是错误的。
                
                preds, embedding = self.model(input_ids, attention_masks, text_lengths)
                
                # (修复: 移除 .squeeze())
                # preds = preds.squeeze()
                
                # (修复: 假设 self.criterion 是函数)
                loss = self.criterion(preds, labels)

                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                self.test_metrics.update('loss', loss.item())
                
                # (关键修复: 保持与 _valid_epoch 一致)
                # 不应在这里使用 argmax。让 metric_ftns 内部处理 logits。
                # preds = preds.argmax(dim=1) 
                
                for met in self.metric_ftns:
                    # 传递 logits (preds) 和 labels
                    self.test_metrics.update(met.__name__, met(preds, labels))

        return self.test_metrics.result()
    
    def _move_batch_to_device(self, data):
        """
        一个辅助函数，用于将数据批次移动到 self.device。
        """
        input_ids, attention_masks, text_lengths, labels = data
        
        input_ids = input_ids.to(self.device)
        if attention_masks is not None:
            attention_masks = attention_masks.to(self.device)
        text_lengths = text_lengths.to(self.device)
        labels = labels.to(self.device)
        
        return input_ids, attention_masks, text_lengths, labels
    
    # (您之前的 _progress 函数没有被调用，tqdm 是更好的替代方案)
    # (如果您仍想使用它，可以在 _train_epoch 内部调用)