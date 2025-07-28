import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import math
from decimal import Decimal, getcontext

# 设置高精度计算
getcontext().prec = 500000  # 设置足够高的精度来处理大数

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 从s4库导入真实的S4Block实现
from s4.models.s4.s4 import S4Block

def decimal_to_binary_no_pad(value):
    """将十进制数转换为不带前导零的二进制字符串"""
    if value == 0:
        return "0"
    binary = bin(value)[2:]  # Remove '0b' prefix
    return binary

def binary_to_decimal(binary_str):
    """将二进制字符串转换为十进制数"""
    return int(binary_str, 2)

class S4ModNModelWithRetrospective(nn.Module):
    """基于S4的模n计算模型，带有回顾机制"""
    def __init__(self, d_model, d_state, n, n_layers=2, dropout=0.1, input_dropout=0.1, seq_len=16):
        super(S4ModNModelWithRetrospective, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n = n
        self.n_layers = n_layers
        # 移除对超大n值的特殊处理，直接计算输出位数
        try:
            self.output_bits = max(1, math.ceil(math.log2(n)))  # 输出二进制位数
        except (ValueError, OverflowError):
            # 对于超大数值，使用位长度计算
            if isinstance(n, int):
                if n > 10**10:  # 对于超大数值使用Decimal
                    decimal_n = Decimal(n)
                    self.output_bits = max(1, int(decimal_n.log10() / Decimal(2).log10()) + 1)
                else:
                    self.output_bits = max(1, n.bit_length())
            else:
                # 如果n是表达式如"2^1000"
                if isinstance(n, str) and n.startswith("2^"):
                    self.output_bits = int(n[2:])
                else:
                    try:
                        self.output_bits = max(1, math.ceil(math.log2(int(n))))
                    except (ValueError, OverflowError):
                        # 最后的备选方案
                        self.output_bits = 64  # 默认值
        self.ptr_bits = math.ceil(math.log2(seq_len))  # 指针位数改为基于序列长度，与数据集保持一致
        self.output_dim = 3 + self.ptr_bits  # 输出维度 (1字符 + 2tag + ptr_bits指针)
        self.seq_len = seq_len
        
        # 输入投影层 - 输入维度为3 (字符0/1 + 2位tag)
        self.input_proj = nn.Linear(3, d_model)
        
        # 多个S4层
        self.s4_layers = nn.ModuleList()
        self.s4_norms = nn.ModuleList()  # 为每个S4层添加独立的LayerNorm
        for i in range(n_layers):
            self.s4_layers.append(S4Block(
                d_model=d_model,
                d_state=d_state,
                l_max=None,  # 不限制序列长度
                channels=1,
                bidirectional=False,
                activation='gelu',
                transposed=False,
                mode='s4',  # 使用完整的S4模型
                init='legs',   # 使用HiPPO-LegS初始化
                dropout=dropout    # 添加dropout
            ))
            self.s4_norms.append(nn.LayerNorm(d_model))
        
        # 输出层 - 生成输出序列的每一位
        # 输出维度为 1字符 + 2tag + ptr_bits指针信息
        self.output_layer = nn.Linear(d_model, self.output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_seq, autoregressive_steps):
        """
        前向传播
        Args:
            input_seq: 输入序列 (batch_size, seq_len, 3)
            autoregressive_steps: 自回归生成的步数
        """
        batch_size, seq_len, _ = input_seq.size()
        
        # 输入投影
        x = self.input_proj(input_seq)  # (batch, seq, d_model)
        
        # 保存编码后的输入序列用于指针访问
        encoded_inputs = x.clone()  # (batch, seq_len, d_model)
        
        # 通过多个S4层处理
        for i in range(self.n_layers):
            residual = x
            x, _ = self.s4_layers[i](x)  # (batch, seq, d_model)
            x = x + residual  # 残差连接
            x = self.s4_norms[i](x)   # 层归一化
        
        # 自回归生成输出
        outputs = []
        # 初始输入为0字符 + tag [0, 1] (输出阶段)
        decoder_input = torch.zeros(batch_size, 1, 3).to(input_seq.device)
        decoder_input[:, :, 1] = 0  # 第一个tag位为0
        decoder_input[:, :, 2] = 1  # 第二个tag位为1，表示输出阶段开始
        
        # 初始化状态缓存和上下文缓存
        states = [None] * self.n_layers
        context = torch.zeros_like(x[:, -1, :])  # 初始上下文
        
        # 需要生成的总步数（输出位 + 指针位）
        # 对于n个输出位，我们需要n个输出 + (n-1)个指针 = 2n-1步
        # 为了避免内存问题，对非常大的输出进行限制
        total_steps = 2 * self.output_bits - 1
        # 限制最大步数以避免内存问题
        max_steps = max(1, min(total_steps, 20001))  # 限制最大步数
        total_steps = max_steps
        
        for i in range(total_steps):
            # 输入投影
            decoder_proj = self.input_proj(decoder_input)  # (batch, 1, d_model)
            
            # 通过多个S4层处理解码器输出，传递状态
            new_states = []
            x_dec = decoder_proj
            for j in range(self.n_layers):
                # 传入前一步的状态
                x_dec, state = self.s4_layers[j](x_dec, state=states[j])  # (batch, 1, d_model)
                new_states.append(state)
                # 添加残差连接和层归一化
                if i == 0:  # 第一次迭代时，需要正确处理残差连接
                    residual = decoder_proj  # (batch, 1, d_model)
                else:
                    residual = x_dec
                x_dec = x_dec + residual  # 残差连接
                x_dec = self.s4_norms[j](x_dec)   # 层归一化
            
            states = new_states  # 更新状态缓存
            
            # 如果是指针步，使用指针访问输入序列
            is_output_step = (i % 2 == 0)  # 偶数步是输出步 (0, 2, 4, ...)
            if not is_output_step:
                # 获取模型生成的指针值
                output = self.sigmoid(self.output_layer(x_dec.squeeze(1) + context))
                ptr_output = output[:, 3:3+self.ptr_bits]
                ptr_index = torch.argmax(ptr_output, dim=1)  # 转换为索引
                
                # 从编码输入中提取指针位置的表示
                ptr_vec = encoded_inputs[torch.arange(batch_size), ptr_index]
                
                # 将指针信息融入上下文
                context = context + ptr_vec
            
            # 生成当前位输出（使用更新后的上下文）
            decoder_out = x_dec.squeeze(1) + context
            
            # 生成当前位输出
            output = self.output_layer(decoder_out)  # (batch, output_dim)
            output = self.sigmoid(output)
            outputs.append(output.unsqueeze(1))  # (batch, 1, output_dim)
            
            # 构造下一个时间步的输入
            is_output_step = (i % 2 == 0)  # 偶数步是输出步 (0, 2, 4, ...)
            
            if is_output_step:
                # 输出步: 取输出的第一位作为下一个输入的字符位
                next_char = (output[:, 0] > 0.5).float().unsqueeze(1)  # (batch, 1)
                # tag [1, 0] 表示指针阶段
                next_tag = torch.zeros(batch_size, 2).to(input_seq.device)
                next_tag[:, 0] = 1  # tag = [1, 0] 表示指针
            else:
                # 指针步: 字符位为0，tag [0, 1] 表示输出阶段
                next_char = torch.zeros(batch_size, 1).to(input_seq.device)  # 字符位为0
                next_tag = torch.zeros(batch_size, 2).to(input_seq.device)
                next_tag[:, 1] = 1  # tag = [0, 1] 表示输出
                
            # 组合输入
            decoder_input = torch.cat([next_char, next_tag], dim=1).unsqueeze(1)  # (batch, 1, 3)
            
        # 将所有输出连接起来
        outputs = torch.cat(outputs, dim=1)  # (batch, total_steps, output_dim)
        return outputs

class ModNDatasetWithRetrospective(Dataset):
    """带有回顾机制的模n数据集"""
    def __init__(self, n_samples=1000, seq_len=16, n=16):
        self.n_samples = n_samples
        self.seq_len = seq_len
        # 处理n为字符串形式的情况
        if isinstance(n, str) and n.startswith("2^"):
            exponent = int(n[2:])
            # 对于任何指数，都使用Decimal计算以避免溢出
            self.n = Decimal(2) ** exponent
            self.output_bits = exponent  # 直接使用指数作为输出位数
        else:
            self.n = n
            # 移除对超大n值的特殊处理，直接计算输出位数
            if isinstance(n, int) and n > 10**10:  # 对于较大数值使用Decimal
                self.output_bits = max(1, int(Decimal(n).log10() / Decimal(2).log10()) + 1)
            else:
                self.output_bits = max(1, math.ceil(math.log2(n)))
        self.ptr_bits = math.ceil(math.log2(seq_len))  # 移除对超大序列长度的限制
        self.data = []
        self.labels = []
        
        # 生成数据
        for _ in range(n_samples):
            # 生成随机二进制序列
            sequence = np.random.randint(0, 2, seq_len)  # 0或1
            
            # 计算模n结果 - 移除对超大数值的特殊处理
            # 使用按位计算避免大数溢出
            # 对于非常大的n值，我们使用Decimal进行精确计算
            if isinstance(self.n, Decimal) or (isinstance(self.n, int) and self.n > 10**10):  # 对于超大数值使用Decimal
                decimal_n = Decimal(self.n)
                modn_result = Decimal(0)
                for bit in sequence:
                    modn_result = (modn_result * 2 + bit) % decimal_n
                modn_result = int(modn_result)
            elif isinstance(self.n, int):
                # 对于较小的数值，使用普通计算
                modn_result = 0
                for bit in sequence:
                    modn_result = (modn_result * 2 + bit) % self.n
            else:
                # 处理其他情况，确保使用Decimal
                decimal_n = Decimal(str(self.n))
                modn_result = Decimal(0)
                for bit in sequence:
                    modn_result = (modn_result * 2 + bit) % decimal_n
                modn_result = int(modn_result)
            
            # 转换为二进制表示（不带前导零）
            binary_result = decimal_to_binary_no_pad(modn_result)
            
            # 对于非常大的输出，限制长度以避免内存问题
            # 但确保至少有1位输出
            max_output_length = max(1, min(len(binary_result), 10000))  # 限制最大输出长度
            binary_result = binary_result[:max_output_length]
            
            # 构造输入序列 (字符 + tag)
            input_seq = []
            for bit in sequence:
                # 输入阶段: 字符 + tag [0, 0]
                input_seq.append([float(bit), 0.0, 0.0])
            
            # 构造输出序列 (字符 + tag + 指针)
            output_seq = []
            # 输出阶段: 字符 + tag [0, 1] + 指针信息
            # 移除对超大n值输出长度的限制
            for i, bit in enumerate(binary_result):
                # 普通输出 tag [0, 1]
                output_item = [float(bit), 0.0, 1.0] + [0.0] * self.ptr_bits  # 指针位置，初始化为0
                output_seq.append(output_item)
                
                # 在每个输出后添加指针，指向下一个需要的位置
                if i < len(binary_result) - 1:  # 不是最后一个输出
                    # 生成指针 (指向序列中的下一个位置)
                    ptr_pos = i + 1
                    
                    # 确保ptr_pos不超过指针位数能表示的最大值
                    ptr_pos = min(ptr_pos, 2**self.ptr_bits - 1)
                    
                    # 转换为二进制
                    ptr_binary = format(ptr_pos, f'0{self.ptr_bits}b')
                    ptr_bits = [float(b) for b in ptr_binary]
                    
                    # tag [1, 0] 表示指针
                    ptr_item = [0.0, 1.0, 0.0] + ptr_bits  # 字符位为0，tag为[1,0]
                    output_seq.append(ptr_item)
            
            self.data.append(np.array(input_seq, dtype=np.float32))
            self.labels.append(np.array(output_seq, dtype=np.float32))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])  # (seq_len, 3)
        y = torch.tensor(self.labels[idx])  # (output_len, 11)
        return x, y

def collate_fn_with_retrospective(batch):
    """自定义批处理函数，处理变长序列"""
    data, labels = zip(*batch)
    data = torch.stack(data, dim=0)  # (batch_size, seq_len, 3)
    
    # 找到最大长度和维度
    max_len = max([label.size(0) for label in labels])
    # 根据实际标签的维度确定输出维度（3 + ptr_bits）
    output_dim = labels[0].size(1)  # 应该是 3 + ptr_bits
    
    # 填充标签到相同长度，使用-1作为填充值
    padded_labels = []
    for label in labels:
        padded_label = torch.full((max_len, output_dim), -1.0)  # 使用-1填充
        padded_label[:label.size(0), :] = label
        padded_labels.append(padded_label)
    
    labels = torch.stack(padded_labels, dim=0)  # (batch_size, max_len, output_dim)
    return data, labels

def train_model(args):
    """训练模型并返回验证准确率"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 计算输出位数 - 统一处理所有n值
    if isinstance(args.n, str) and args.n.startswith("2^"):
        output_bits = int(args.n[2:])  # 直接使用指数作为输出位数
        n_display = args.n
        exponent = int(args.n[2:])
        # 对于任何指数，都使用Decimal计算以避免溢出
        n_value = Decimal(2) ** exponent
    else:
        n_value = int(args.n) if isinstance(args.n, str) else args.n
        try:
            output_bits = max(1, math.ceil(math.log2(n_value)))
        except (ValueError, OverflowError):
            # 对于超大数值，使用Decimal进行计算
            decimal_n = Decimal(n_value)
            output_bits = max(1, int(decimal_n.log10() / Decimal(2).log10()) + 1)
        n_display = f"2^{int(math.log2(n_value))}" if (isinstance(n_value, int) and n_value > 10**6) else n_value
        
    ptr_bits = math.ceil(math.log2(args.seq_len))  # 移除对超大序列长度的限制
    print(f"模数n={n_display}, 输出位数={output_bits}, 指针位数={ptr_bits}")
    
    # 创建数据集
    train_dataset = ModNDatasetWithRetrospective(args.n_train, args.seq_len, n_value)
    val_dataset = ModNDatasetWithRetrospective(args.n_val, args.seq_len, n_value)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn_with_retrospective)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn_with_retrospective)
    
    # 创建模型 - 使用实际的序列长度，而不是默认的args.seq_len
    # 通过设置seq_len为实际值，支持超大序列长度
    model = S4ModNModelWithRetrospective(
        d_model=args.d_model, 
        d_state=args.d_state, 
        n=n_value, 
        n_layers=args.n_layers,
        dropout=getattr(args, 'dropout', 0.1),
        input_dropout=getattr(args, 'input_dropout', 0.1),
        seq_len=train_dataset.seq_len  # 使用数据集的实际序列长度
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    
    # 使用BCELoss损失函数
    criterion = nn.BCELoss()
    # 添加权重衰减增强正则化
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 添加梯度裁剪参数
    gradient_clip_value = args.gradient_clip if hasattr(args, 'gradient_clip') else 1.0
    
    # 添加学习率预热相关参数
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    warmup_lr = getattr(args, 'warmup_lr', 1e-6)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'grad_norm': [],  # 添加梯度范数记录
        'lr': []  # 添加学习率记录
    }
    
    best_acc = 0.0
    best_model = None
    best_epoch = 0  # 记录达到最佳准确率的epoch
    
    # 移除对超大数值训练轮数的限制
    epochs = args.epochs
    
    for epoch in range(epochs):
        # 学习率预热
        if epoch < warmup_epochs:
            # 线性预热
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = warmup_lr + (args.lr - warmup_lr) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif epoch == warmup_epochs:
            # 预热结束后恢复到原始学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        grad_norm_sum = 0.0  # 用于计算平均梯度范数
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # 总步数为 2 * output_bits - 1 (输出位 + 指针位)
            # 为了避免内存问题，对非常大的输出进行限制
            total_steps = 2 * output_bits - 1
            # 限制最大步数以避免内存问题
            total_steps = max(1, min(total_steps, 20001))
                
            outputs = model(inputs, total_steps)  # (batch, total_steps, output_dim)
            
            # 创建掩码，忽略填充值-1
            # 检查整个向量的第一个元素是否为-1，表示整个位置是填充的
            mask = (labels[:, :total_steps, 0] != -1)  # 将labels裁剪到total_steps长度
            
            # 重新组织labels以匹配outputs的维度
            labels_reshaped = labels[:, :total_steps, :]  # 取全部维度进行比较
            
            # 确保outputs和labels_reshaped具有相同的形状
            # 如果形状不匹配，则进行调整
            if outputs.shape != labels_reshaped.shape:
                min_batch = min(outputs.shape[0], labels_reshaped.shape[0])
                min_steps = min(outputs.shape[1], labels_reshaped.shape[1])
                min_dim = min(outputs.shape[2], labels_reshaped.shape[2])
                outputs = outputs[:min_batch, :min_steps, :min_dim]
                labels_reshaped = labels_reshaped[:min_batch, :min_steps, :min_dim]
                mask = mask[:min_batch, :min_steps]
            
            # 扩展mask以匹配labels_reshaped的形状
            mask_expanded = mask.unsqueeze(-1).expand_as(labels_reshaped)
            
            # 计算损失，只考虑非填充值
            # 使用masked_select来选择非填充的数据
            loss = criterion(outputs.masked_select(mask_expanded),
                            labels_reshaped.masked_select(mask_expanded))
            loss.backward()
            
            # 添加梯度监控
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1. / 2)
            grad_norm_sum += grad_norm
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算准确率：只考虑非填充值
            predicted = (outputs > 0.5).float()
            total += labels_reshaped[mask_expanded].numel()  # 只计算非填充值
            correct += (predicted[mask_expanded] == labels_reshaped[mask_expanded]).sum().item()
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = grad_norm_sum / len(train_loader)  # 计算平均梯度范数
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['grad_norm'].append(avg_grad_norm)  # 记录梯度范数
        history['lr'].append(current_lr)  # 记录当前学习率
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # 总步数为 2 * output_bits - 1 (输出位 + 指针位)
                total_steps = 2 * output_bits - 1
                outputs = model(inputs, total_steps)  # (batch, total_steps, output_dim)
                
                # 创建掩码，忽略填充值-1
                mask = (labels[:, :total_steps, 0] != -1)  # 使用第一个输出位判断是否为填充值
                
                # 重新组织labels以匹配outputs的维度
                labels_reshaped = labels[:, :total_steps, :]  # 取全部维度进行比较
                
                # 确保outputs和labels_reshaped具有相同的形状
                # 如果形状不匹配，则进行调整
                if outputs.shape != labels_reshaped.shape:
                    min_batch = min(outputs.shape[0], labels_reshaped.shape[0])
                    min_steps = min(outputs.shape[1], labels_reshaped.shape[1])
                    min_dim = min(outputs.shape[2], labels_reshaped.shape[2])
                    outputs = outputs[:min_batch, :min_steps, :min_dim]
                    labels_reshaped = labels_reshaped[:min_batch, :min_steps, :min_dim]
                    mask = mask[:min_batch, :min_steps]
                
                # 扩展mask以匹配labels_reshaped的形状
                mask_expanded = mask.unsqueeze(-1).expand_as(labels_reshaped)
                
                # 计算损失，只考虑非填充值
                loss = criterion(outputs.masked_select(mask_expanded),
                                labels_reshaped.masked_select(mask_expanded))
                
                val_loss += loss.item()
                
                # 计算准确率：只考虑非填充值
                predicted = (outputs > 0.5).float()
                total += labels_reshaped[mask_expanded].numel()  # 只计算非填充值
                correct += (predicted[mask_expanded] == labels_reshaped[mask_expanded]).sum().item()
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # 更新学习率（预热结束后才使用调度器）
        if epoch >= warmup_epochs:
            scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict().copy()
            best_epoch = epoch + 1  # 记录达到最佳准确率的epoch
        
        # 打印包含梯度信息的训练日志
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}% | Grad Norm: {avg_grad_norm:.4f} | LR: {current_lr:.6f}')
    
    # 保存最佳模型
    if best_model is not None:
        # 对于超大n值，使用表达式形式作为文件名一部分
        if isinstance(n_expr, str) and n_expr.startswith("2^"):
            n_for_filename = n_expr  # 直接使用表达式形式
        elif isinstance(n_value, Decimal) or (isinstance(n_value, int) and n_value > 10**6):
            # 对于超大数值，尝试使用log2计算，但如果失败则使用表达式
            try:
                if isinstance(n_value, Decimal):
                    n_for_filename = f"2^{int(n_value.log10() / Decimal(2).log10())}"
                else:
                    n_for_filename = f"2^{int(math.log2(n_value))}"
            except (ValueError, OverflowError):
                n_for_filename = str(n_value)
        else:
            n_for_filename = n_value
        torch.save(best_model, f's4_retro_best_model_n{n_for_filename}.pth')
        print(f"\n训练完成！")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        print(f"达到最佳准确率的Epoch: {best_epoch}")
    
    # 加载最佳模型
    if isinstance(n_expr, str) and n_expr.startswith("2^"):
        n_for_filename = n_expr  # 直接使用表达式形式
    elif isinstance(n_value, Decimal) or (isinstance(n_value, int) and n_value > 10**6):
        # 对于超大数值，尝试使用log2计算，但如果失败则使用表达式
        try:
            if isinstance(n_value, Decimal):
                n_for_filename = f"2^{int(n_value.log10() / Decimal(2).log10())}"
            else:
                n_for_filename = f"2^{int(math.log2(n_value))}"
        except (ValueError, OverflowError):
            n_for_filename = str(n_value)
    else:
        n_for_filename = n_value
        
    model = S4ModNModelWithRetrospective(args.d_model, args.d_state, n_value, args.n_layers, seq_len=args.seq_len).to(device)
    model.load_state_dict(torch.load(f's4_retro_best_model_n{n_for_filename}.pth'))
    
    # 绘制训练曲线
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 4, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.subplot(1, 4, 3)
    plt.plot(history['grad_norm'], label='Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.title('Gradient Norm Curve')
    
    plt.subplot(1, 4, 4)
    plt.plot(history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Curve')
    plt.yscale('log')
    # 修复类型错误，使用已解析的n_value而不是args.n
    n_for_filename = f"2^{int(math.log2(n_value))}" if n_value > 10**6 else n_value
    plt.savefig(f's4_retro_training_curves_n{n_for_filename}.png')
    plt.close()
    
    return best_acc, history, best_epoch

def test_model(args):
    """测试模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 计算输出位数 - 移除对超大n值的特殊处理
    if isinstance(args.n, str) and args.n.startswith("2^"):
        output_bits = int(args.n[2:])  # 直接使用指数作为输出位数
        n_display = args.n
        exponent = int(args.n[2:])
        # 对于任何指数，都使用Decimal计算以避免溢出
        n_value = Decimal(2) ** exponent
    else:
        n_value = int(args.n) if isinstance(args.n, str) else args.n
        try:
            output_bits = max(1, math.ceil(math.log2(n_value)))
        except (ValueError, OverflowError):
            # 对于超大数值，使用位长度计算
            if isinstance(n_value, int) and n_value > 10**10:
                decimal_n = Decimal(n_value)
                output_bits = max(1, int(decimal_n.log10() / Decimal(2).log10()) + 1)
            else:
                output_bits = max(1, n_value.bit_length())
        n_display = f"2^{int(math.log2(n_value))}" if (isinstance(n_value, int) and n_value > 10**6) else n_value
        
    ptr_bits = math.ceil(math.log2(args.seq_len))  # 移除对超大序列长度的限制
    print(f"模数n={n_display}, 输出位数={output_bits}, 指针位数={ptr_bits}")
    
    # 创建测试数据集
    test_dataset = ModNDatasetWithRetrospective(n_samples=args.n_test, seq_len=args.seq_len, n=n_value)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=collate_fn_with_retrospective)
    
    # 加载模型
    model = S4ModNModelWithRetrospective(
        d_model=args.d_model, 
        d_state=args.d_state, 
        n=n_value, 
        n_layers=args.n_layers,
        dropout=getattr(args, 'dropout', 0.1),
        input_dropout=getattr(args, 'input_dropout', 0.1),
        seq_len=args.seq_len  # 传递序列长度参数
    ).to(device)
    n_for_filename = f"2^{int(math.log2(n_value))}" if n_value > 10**6 else n_value
    model.load_state_dict(torch.load(f's4_retro_best_model_n{n_for_filename}.pth'))
    model.eval()
    
    criterion = nn.BCELoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 总步数为 2 * output_bits - 1 (输出位 + 指针位)
            # 为了避免内存问题，对非常大的输出进行限制
            total_steps = 2 * output_bits - 1
            # 限制最大步数以避免内存问题
            total_steps = max(1, min(total_steps, 20001))
            outputs = model(inputs, total_steps)  # (batch, total_steps, output_dim)
            
            # 创建掩码，忽略填充值-1
            # 限制labels到total_steps长度，并检查第一个元素是否为-1
            mask = (labels[:, :total_steps, 0] != -1)  # (batch, total_steps)
            
            # 重新组织labels以匹配outputs的维度
            labels_reshaped = labels[:, :total_steps, :]  # 取全部维度进行比较
            
            # 确保outputs和labels_reshaped具有相同的形状
            # 如果形状不匹配，则进行调整
            if outputs.shape != labels_reshaped.shape:
                min_batch = min(outputs.shape[0], labels_reshaped.shape[0])
                min_steps = min(outputs.shape[1], labels_reshaped.shape[1])
                min_dim = min(outputs.shape[2], labels_reshaped.shape[2])
                outputs = outputs[:min_batch, :min_steps, :min_dim]
                labels_reshaped = labels_reshaped[:min_batch, :min_steps, :min_dim]
                mask = mask[:min_batch, :min_steps]
            
            # 扩展mask以匹配labels_reshaped的形状
            mask_expanded = mask.unsqueeze(-1).expand_as(labels_reshaped)
            
            # 计算损失，只考虑非填充值
            loss = criterion(outputs.masked_select(mask_expanded),
                           labels_reshaped.masked_select(mask_expanded))
            
            test_loss += loss.item()
            
            # 计算准确率：只考虑非填充值
            predicted = (outputs > 0.5).float()
            total += labels_reshaped[mask].numel()  # 只计算非填充值
            correct += (predicted[mask] == labels_reshaped[mask]).sum().item()
            
            # 收集预测结果用于计算准确率
            # 只收集输出步的预测结果（偶数索引位置）
            pred_binary = (predicted.cpu().numpy() > 0.5).astype(int)
            label_binary = labels_reshaped.cpu().numpy().astype(int)
            
            for i in range(pred_binary.shape[0]):
                # 提取输出步的结果（偶数索引位置：0, 2, 4, ...）
                pred_output_bits = []
                label_output_bits = []
                
                # 移除对最大output_bits的限制，处理所有可能的输出位
                for j in range(len(label_binary[i])):  # 处理所有可能的输出位
                    if j % 2 == 0:  # 输出步 (0, 2, 4, ...)
                        # 确保不是填充值（检查整个向量是否为-1）
                        if not (label_binary[i][j] == -1).all():  
                            pred_output_bits.append(str(int(pred_binary[i][j][0])))
                            label_output_bits.append(str(int(label_binary[i][j][0])))
                
                # 转换为十进制进行比较
                try:
                    pred_str = ''.join(pred_output_bits)
                    label_str = ''.join(label_output_bits)
                    pred_decimal = binary_to_decimal(pred_str) if pred_str else 0
                    label_decimal = binary_to_decimal(label_str) if label_str else 0
                    all_preds.append(pred_decimal)
                    all_labels.append(label_decimal)
                except ValueError:
                    # 处理转换错误的情况
                    all_preds.append(0)
                    all_labels.append(0)
    
    test_acc = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    # 计算完全匹配的准确率
    exact_match = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    exact_match_acc = 100. * exact_match / len(all_preds) if all_preds else 0
    
    # 打印详细报告
    print(f"\n{'='*50}")
    print(f"测试结果 (n={n_display}):")
    print(f"测试损失: {avg_test_loss:.4f}")
    print(f"位准确率: {test_acc:.2f}%")
    print(f"数值准确率: {exact_match_acc:.2f}%")
    print(f"样本总数: {total}")
    print(f"正确预测位数: {correct}")
    print('='*50)
    
    # 绘制混淆矩阵
    if all_preds and all_labels:
        # 限制矩阵大小以提高可读性
        max_val = min(n_value - 1, 15)  # 限制最大显示值
        cm_labels = list(range(min(n_value, 16)))
        
        # 截断值到有效范围
        truncated_preds = [min(p, max_val) for p in all_preds]
        truncated_labels = [min(l, max_val) for l in all_labels]
        
        cm = confusion_matrix(truncated_labels, truncated_preds, labels=cm_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[str(i) for i in cm_labels], 
                    yticklabels=[str(i) for i in cm_labels])
        plt.xlabel('Predicted')
        n_display = f"2^{int(math.log2(n_value))}" if n_value > 10**6 else n_value
        plt.ylabel('True')
        plt.title(f'测试混淆矩阵 (n={n_display})')
        plt.savefig(f's4_retro_test_confusion_matrix_n{n_value}.png')
        plt.close()
    
    return test_acc, exact_match_acc

def main():
    # 开始计时
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='模n二进制序列处理任务 - 带回顾机制的S4模型')
    
    # 数据参数
    parser.add_argument('--n_train', type=int, default=8192, help='训练样本数')
    parser.add_argument('--n_val', type=int, default=1024, help='验证样本数')
    parser.add_argument('--n_test', type=int, default=2048, help='测试样本数')
    parser.add_argument('--seq_len', type=int, default=16, help='输入序列长度')
    parser.add_argument('--n', type=str, default="16", help='模n值，支持超大数值如2^131072')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=4, help='S4模型维度')
    parser.add_argument('--d_state', type=int, default=4, help='S4状态维度')
    parser.add_argument('--n_layers', type=int, default=1, help='S4层数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='S4层dropout率')
    parser.add_argument('--input_dropout', type=float, default=0.0, help='输入层dropout率')
    parser.add_argument('--output_dropout', type=float, default=0.2, help='输出层dropout率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=256, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.003, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='权重衰减(L2正则化)')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='梯度裁剪值')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='学习率预热轮数')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='预热初始学习率')
    args = parser.parse_args()
    
    # 解析n值，支持超大数值表达式
    n_expr = args.n
    try:
        if args.n.startswith("2^"):
            # 处理2的幂次形式
            exponent = int(args.n[2:])
            # 对于任何指数，都使用Decimal计算以避免溢出
            n_value = Decimal(2) ** exponent
        else:
            # 处理普通数值
            n_value = int(args.n)
    except ValueError:
        print(f"无法解析n值: {args.n}，使用默认值16")
        n_value = 16
        n_expr = "16"
    
    # 增加整数转字符串的限制，以支持超大整数的处理
    import sys
    sys.set_int_max_str_digits(10**6)
    
    # 打印信息时使用表达式而不是数值，避免超大整数转换问题
    print(f"开始模{n_expr}二进制序列处理任务实验(带回顾机制)...")
    print(f"配置: seq_len={args.seq_len}, d_model={args.d_model}, d_state={args.d_state}, n_layers={args.n_layers}, n={n_expr}")
    print(f"学习率: {args.lr}, 权重衰减: {args.weight_decay}, 梯度裁剪值: {args.gradient_clip}")
    print(f"Dropout配置: S4层={args.dropout}, 输入层={args.input_dropout}, 输出层={args.output_dropout}")
    print(f"学习率预热: {args.warmup_epochs} 轮, 预热起始学习率: {args.warmup_lr}")
    
    # 确保args.n保持为字符串类型，以便模型和数据集正确处理
    args.n = str(n_expr)  # 始终保持args.n为字符串类型
    
    # 确保args.n保持为字符串类型，以便模型和数据集正确处理
    args.n = str(n_expr)  # 始终保持args.n为字符串类型
    
    # 训练模型
    best_acc, history, best_epoch = train_model(args)
    
    # 测试模型
    bit_acc, exact_match_acc = test_model(args)
    
    # 结束计时并输出
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")
    
    # 保存配置和结果
    # 使用表达式形式而不是实际数值，避免大数转换问题
    if isinstance(n_expr, str) and n_expr.startswith("2^"):
        n_for_filename = n_expr  # 直接使用表达式形式
    elif isinstance(n_value, Decimal) or (isinstance(n_value, int) and n_value > 10**6):
        # 对于超大数值，尝试使用log2计算，但如果失败则使用表达式
        try:
            if isinstance(n_value, Decimal):
                n_for_filename = f"2^{int(n_value.log10() / Decimal(2).log10())}"
            else:
                n_for_filename = f"2^{int(math.log2(n_value))}"
        except (ValueError, OverflowError):
            n_for_filename = str(n_value)
    else:
        n_for_filename = n_value
        
    with open(f's4_retro_results_n{n_for_filename}.txt', 'w') as f:
        f.write(f"实验配置:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("\n结果:\n")
        f.write(f"最佳验证准确率: {best_acc:.2f}%\n")
        f.write(f"测试位准确率: {bit_acc:.2f}%\n")
        f.write(f"测试数值准确率: {exact_match_acc:.2f}%\n")
        f.write(f"达到最佳准确率的Epoch数: {best_epoch}\n")
        f.write(f"总运行时间: {elapsed_time:.2f} 秒\n")

if __name__ == "__main__":
    main()