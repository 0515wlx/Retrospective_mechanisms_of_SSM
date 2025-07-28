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

class S4ModNModel(nn.Module):
    """基于S4的模n计算模型"""
    def __init__(self, d_model, d_state, n, n_layers=2, dropout=0.1, input_dropout=0.0, output_dropout=0.2):
        super(S4ModNModel, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n = n
        self.n_layers = n_layers
        self.output_bits = max(1, math.ceil(math.log2(n)))  # 输出二进制位数
        
        # 输入投影层
        self.input_proj = nn.Linear(1, d_model)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        
        # 多个S4层
        self.s4_layers = nn.ModuleList()
        self.s4_dropouts = nn.ModuleList()  # 为每个S4层添加独立的dropout
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
            self.s4_dropouts.append(nn.Dropout(dropout))
        
        # 层归一化
        self.layer_norms = nn.ModuleList()
        for i in range(n_layers):
            self.layer_norms.append(nn.LayerNorm(d_model))
        
        # 输出层 - 生成输出序列的每一位
        self.output_layer = nn.Linear(d_model, 1)
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout > 0 else nn.Identity()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_seq, autoregressive_steps):
        """
        前向传播
        Args:
            input_seq: 输入序列 (batch_size, seq_len, 1)
            autoregressive_steps: 自回归生成的步数
        """
        batch_size, seq_len, _ = input_seq.size()
        
        # 输入投影
        x = self.input_proj(input_seq)  # (batch, seq, d_model)
        x = self.input_dropout(x)
        
        # 通过多个S4层处理
        for i in range(self.n_layers):
            residual = x
            x, _ = self.s4_layers[i](x)  # (batch, seq, d_model)
            x = self.s4_dropouts[i](x)   # 应用dropout
            x = x + residual  # 残差连接
            x = self.layer_norms[i](x)   # 层归一化
        
        # 自回归生成输出
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, 1).to(input_seq.device)  # 初始输入为0
        
        # 使用最后一个时间步的表示作为上下文
        context = x[:, -1, :]  # (batch, d_model)
        
        # 初始化状态缓存
        states = [None] * self.n_layers
        
        for i in range(autoregressive_steps):
            # 使用上下文和解码器输入
            # 这里我们使用上下文向量作为额外信息
            decoder_out = decoder_input
            
            # 通过多个S4层处理解码器输出，传递状态
            new_states = []
            for j in range(self.n_layers):
                # 传入前一步的状态
                decoder_out, state = self.s4_layers[j](decoder_out, state=states[j])  # (batch, 1, d_model)
                new_states.append(state)
                decoder_out = self.s4_dropouts[j](decoder_out)   # 应用dropout
                # 添加残差连接和层归一化
                if i == 0:  # 第一次迭代时，decoder_out需要与decoder_input对齐维度
                    residual = decoder_input.repeat(1, 1, self.d_model)
                else:
                    residual = decoder_out
                decoder_out = decoder_out + residual  # 残差连接
                decoder_out = self.layer_norms[j](decoder_out)   # 层归一化
            
            states = new_states  # 更新状态缓存
            
            decoder_out = decoder_out.squeeze(1) + context  # (batch, d_model)
            
            # 生成当前位输出
            decoder_out = self.output_dropout(decoder_out)  # 应用输出dropout
            output_bit = self.output_layer(decoder_out)  # (batch, 1)
            output_bit = self.sigmoid(output_bit)
            outputs.append(output_bit.unsqueeze(1))  # (batch, 1, 1)
            
            # 将当前输出作为下一时刻的输入
            decoder_input = output_bit.unsqueeze(1)  # (batch, 1, 1)
            
        # 将所有输出连接起来
        outputs = torch.cat(outputs, dim=1)  # (batch, autoregressive_steps, 1)
        return outputs

class RNNModNModel(nn.Module):
    """基于RNN的模n计算模型"""
    def __init__(self, hidden_size, n):
        super(RNNModNModel, self).__init__()
        self.hidden_size = hidden_size
        self.n = n
        self.output_bits = max(1, math.ceil(math.log2(n)))  # 输出二进制位数
        
        # 输入层 (input_size=1 for binary bits)
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        
        # 输出层 - 生成输出序列的每一位
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_seq, autoregressive_steps):
        """
        前向传播
        Args:
            input_seq: 输入序列 (batch_size, seq_len, 1)
            autoregressive_steps: 自回归生成的步数
        """
        batch_size, seq_len, _ = input_seq.size()
        
        # 阶段1：读取输入序列
        rnn_out, hidden = self.rnn(input_seq)  # rnn_out: (batch, seq_len, hidden_size)
        
        # 阶段2：自回归生成输出
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, 1).to(input_seq.device)  # 初始输入为0
        
        for i in range(autoregressive_steps):
            # 使用上一时刻的隐藏状态作为当前输入的上下文
            decoder_out, hidden = self.rnn(decoder_input, hidden)
            
            # 生成当前位输出
            output_bit = self.output_layer(decoder_out)  # (batch, 1, 1)
            output_bit = self.sigmoid(output_bit)
            outputs.append(output_bit)
            
            # 将当前输出作为下一时刻的输入
            decoder_input = output_bit
            
        # 将所有输出连接起来
        outputs = torch.cat(outputs, dim=1)  # (batch, autoregressive_steps, 1)
        return outputs

class ModNDataset(Dataset):
    """模n数据集"""
    def __init__(self, n_samples=1000, seq_len=16, n=16):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n = n
        self.output_bits = max(1, math.ceil(math.log2(n)))
        self.data = []
        self.labels = []
        
        # 生成数据
        for _ in range(n_samples):
            # 生成随机二进制序列
            sequence = np.random.randint(0, 2, seq_len)  # 0或1
            binary_str = ''.join([str(bit) for bit in sequence])
            
            # 计算模n结果
            decimal_value = binary_to_decimal(binary_str)
            modn_result = decimal_value % n
            
            # 转换为二进制表示（不带前导零）
            binary_result = decimal_to_binary_no_pad(modn_result)
            
            # 确保输出长度一致，但不添加前导零
            self.data.append(sequence.astype(np.float32))
            
            # 将目标二进制字符串转换为浮点数数组（0或1）
            label_bits = [float(bit) for bit in binary_result]
            self.labels.append(label_bits)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).unsqueeze(-1)  # (seq_len, 1)
        # 对于标签，我们需要处理变长序列
        y = torch.tensor(self.labels[idx], dtype=torch.float32)  # (output_bits,)
        return x, y

def collate_fn(batch):
    """自定义批处理函数，处理变长序列"""
    data, labels = zip(*batch)
    data = torch.stack(data, dim=0)  # (batch_size, seq_len, 1)
    
    # 找到最大长度
    max_len = max([label.size(0) for label in labels])
    
    # 填充标签到相同长度，使用-1作为填充值
    padded_labels = []
    for label in labels:
        padded_label = torch.full((max_len,), -1.0)  # 使用-1填充
        padded_label[:label.size(0)] = label
        padded_labels.append(padded_label)
    
    labels = torch.stack(padded_labels, dim=0)  # (batch_size, max_len)
    return data, labels

def train_model(args):
    """训练模型并返回验证准确率"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 计算输出位数
    output_bits = max(1, math.ceil(math.log2(args.n)))
    print(f"模数n={args.n}, 输出位数={output_bits}")
    
    # 创建数据集
    train_dataset = ModNDataset(args.n_train, args.seq_len, args.n)
    val_dataset = ModNDataset(args.n_val, args.seq_len, args.n)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    model = S4ModNModel(
        d_model=args.d_model, 
        d_state=args.d_state, 
        n=args.n, 
        n_layers=args.n_layers,
        dropout=getattr(args, 'dropout', 0.1),
        input_dropout=getattr(args, 'input_dropout', 0.0),
        output_dropout=getattr(args, 'output_dropout', 0.2)
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
    
    for epoch in range(args.epochs):
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
            outputs = model(inputs, output_bits)  # (batch, output_bits, 1)
            outputs = outputs.squeeze(-1)  # (batch, output_bits)
            
            # 创建掩码，忽略填充值-1
            mask = (labels != -1)
            # 计算损失，只考虑非填充值
            loss = criterion(outputs[mask], labels[mask])
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
            total += labels[mask].numel()  # 只计算非填充值
            correct += (predicted[mask] == labels[mask]).sum().item()
        
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
                outputs = model(inputs, output_bits)  # (batch, output_bits, 1)
                outputs = outputs.squeeze(-1)  # (batch, output_bits)
                
                # 创建掩码，忽略填充值-1
                mask = (labels != -1)
                # 计算损失，只考虑非填充值
                loss = criterion(outputs[mask], labels[mask])
                
                val_loss += loss.item()
                
                # 计算准确率：只考虑非填充值
                predicted = (outputs > 0.5).float()
                total += labels[mask].numel()  # 只计算非填充值
                correct += (predicted[mask] == labels[mask]).sum().item()
        
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
        torch.save(best_model, f's4_best_model_n{args.n}.pth')
        print(f"\n训练完成！")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        print(f"达到最佳准确率的Epoch: {best_epoch}")
    
    # 加载最佳模型
    model = S4ModNModel(args.d_model, args.d_state, args.n, args.n_layers).to(device)
    model.load_state_dict(torch.load(f's4_best_model_n{args.n}.pth'))
    
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
    plt.savefig(f's4_training_curves_n{args.n}.png')
    plt.close()
    
    return best_acc, history, best_epoch

def test_model(args):
    """测试模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试模型 (n={args.n})...")
    
    # 计算输出位数
    output_bits = max(1, math.ceil(math.log2(args.n)))
    print(f"模数n={args.n}, 输出位数={output_bits}")
    
    # 创建测试数据集
    test_dataset = ModNDataset(n_samples=args.n_test, seq_len=args.seq_len, n=args.n)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 加载模型
    model = S4ModNModel(
        d_model=args.d_model, 
        d_state=args.d_state, 
        n=args.n, 
        n_layers=args.n_layers,
        dropout=getattr(args, 'dropout', 0.1),
        input_dropout=getattr(args, 'input_dropout', 0.0),
        output_dropout=getattr(args, 'output_dropout', 0.2)
    ).to(device)
    model.load_state_dict(torch.load(f's4_best_model_n{args.n}.pth'))
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
            
            outputs = model(inputs, output_bits)  # (batch, output_bits, 1)
            outputs = outputs.squeeze(-1)  # (batch, output_bits)
            
            # 创建掩码，忽略填充值-1
            mask = (labels != -1)
            # 计算损失，只考虑非填充值
            loss = criterion(outputs[mask], labels[mask])
            
            test_loss += loss.item()
            
            # 计算准确率：只考虑非填充值
            predicted = (outputs > 0.5).float()
            total += labels[mask].numel()  # 只计算非填充值
            correct += (predicted[mask] == labels[mask]).sum().item()
            
            # 收集预测结果用于计算准确率
            pred_binary = (predicted.cpu().numpy() > 0.5).astype(int)
            label_binary = labels.cpu().numpy().astype(int)
            
            for i in range(pred_binary.shape[0]):
                # 将二进制数组转换为字符串
                pred_str = ''.join([str(int(b)) for b in pred_binary[i] if b in [0, 1]])
                label_str = ''.join([str(int(b)) for b in label_binary[i] if b in [0, 1]])
                
                # 转换为十进制进行比较
                try:
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
    print(f"测试结果 (n={args.n}):")
    print(f"测试损失: {avg_test_loss:.4f}")
    print(f"位准确率: {test_acc:.2f}%")
    print(f"数值准确率: {exact_match_acc:.2f}%")
    print(f"样本总数: {total}")
    print(f"正确预测位数: {correct}")
    print('='*50)
    
    # 绘制混淆矩阵
    if all_preds and all_labels:
        # 限制矩阵大小以提高可读性
        max_val = min(args.n - 1, 15)  # 限制最大显示值
        cm_labels = list(range(min(args.n, 16)))
        
        # 截断值到有效范围
        truncated_preds = [min(p, max_val) for p in all_preds]
        truncated_labels = [min(l, max_val) for l in all_labels]
        
        cm = confusion_matrix(truncated_labels, truncated_preds, labels=cm_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[str(i) for i in cm_labels], 
                    yticklabels=[str(i) for i in cm_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'测试混淆矩阵 (n={args.n})')
        plt.savefig(f's4_test_confusion_matrix_n{args.n}.png')
        plt.close()
    
    return test_acc, exact_match_acc

def main():
    # 开始计时
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='模n二进制序列处理任务 - S4模型')
    
    # 数据参数
    parser.add_argument('--n_train', type=int, default=8192, help='训练样本数')
    parser.add_argument('--n_val', type=int, default=1024, help='验证样本数')
    parser.add_argument('--n_test', type=int, default=2048, help='测试样本数')
    parser.add_argument('--seq_len', type=int, default=16, help='输入序列长度')
    parser.add_argument('--n', type=int, default=65536, help='模n值')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=4, help='S4模型维度')
    parser.add_argument('--d_state', type=int, default=4, help='S4状态维度')
    parser.add_argument('--n_layers', type=int, default=1, help='S4层数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='S4层dropout率')
    parser.add_argument('--input_dropout', type=float, default=0.0, help='输入层dropout率')
    parser.add_argument('--output_dropout', type=float, default=0.2, help='输出层dropout率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=128, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.003, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减(L2正则化)')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='梯度裁剪值')
    parser.add_argument('--warmup_epochs', type=int, default=4, help='学习率预热轮数')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='预热初始学习率')
    args = parser.parse_args()
    
    print(f"开始模{args.n}二进制序列处理任务实验...")
    print(f"配置: seq_len={args.seq_len}, d_model={args.d_model}, d_state={args.d_state}, n_layers={args.n_layers}, n={args.n}")
    print(f"学习率: {args.lr}, 权重衰减: {args.weight_decay}, 梯度裁剪值: {args.gradient_clip}")
    print(f"Dropout配置: S4层={args.dropout}, 输入层={args.input_dropout}, 输出层={args.output_dropout}")
    print(f"学习率预热: {args.warmup_epochs} 轮, 预热起始学习率: {args.warmup_lr}")
    
    # 训练模型
    best_acc, history, best_epoch = train_model(args)
    
    # 测试模型
    bit_acc, exact_match_acc = test_model(args)
    
    # 结束计时并输出
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")
    
    # 保存配置和结果
    with open(f's4_results_n{args.n}.txt', 'w') as f:
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