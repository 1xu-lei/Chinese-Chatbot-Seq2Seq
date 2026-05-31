# -*- coding: utf-8 -*-
"""
聊天机器人训练模块
优化：Label Smoothing / 学习率预热+余弦退火 / 权重衰减
"""
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from model import Seq2SeqChatbot


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    with open(config.IDS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(config.VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    embedding_matrix = np.load(config.EMBEDDING_FILE)

    source = np.array(data['source'])
    target = np.array(data['target'])
    src_lengths = np.array(data['source_lengths'])
    tgt_lengths = np.array(data['target_lengths'])
    vocab = data['vocab']
    word2id = vocab_data['word2id']
    id2word = {int(k): v for k, v in vocab_data['id2word'].items()}

    return source, target, src_lengths, tgt_lengths, vocab, word2id, id2word, embedding_matrix


def create_dataloader(source, target, src_lengths, tgt_lengths, batch_size):
    src_tensor = torch.LongTensor(source)
    tgt_tensor = torch.LongTensor(target)
    src_len_tensor = torch.LongTensor(src_lengths)
    tgt_len_tensor = torch.LongTensor(tgt_lengths)
    dataset = TensorDataset(src_tensor, tgt_tensor, src_len_tensor, tgt_len_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader


def get_lr(epoch, num_epochs, base_lr, warmup_epochs):
    # 学习率调度：预热 + 余弦退火
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_model(model, train_loader, num_epochs, learning_rate, word2id, id2word,
                source, target, src_lengths, tgt_lengths, vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Label Smoothing 交叉熳
    criterion = nn.CrossEntropyLoss(
        ignore_index=config.PAD_ID,
        label_smoothing=config.LABEL_SMOOTHING
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )

    # 加载已有checkpoint（自动检测词典是否匹配）
    start_epoch = 0
    losses = []
    if os.path.exists(config.MODEL_FILE):
        checkpoint = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)
        # 检查词典大小是否匹配
        old_emb_size = checkpoint['model_state_dict']['encoder.embedding.weight'].shape[0]
        if old_emb_size != vocab_size:
            print(f"[训练] 词典大小变化 ({old_emb_size} -> {vocab_size})，忽略旧checkpoint")
            os.remove(config.MODEL_FILE)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            losses = checkpoint.get('losses', [])
            print(f"[训练] 从epoch {start_epoch} 继续训练")
    if start_epoch == 0:
        print("[训练] 重新开始训练模型")

    print(f"[训练] 使用设备: {device}")
    print(f"[训练] 总轮次: {num_epochs}, 学习率: {learning_rate}")
    print(f"[训练] 语料大小: {len(source)}, 词典大小: {len(word2id)}")
    print(f"[训练] Label Smoothing: {config.LABEL_SMOOTHING}")
    print(f"[训练] Beam Size: {config.BEAM_SIZE}")
    print(f"{'='*60}")

    model.train()

    for epoch in range(start_epoch, num_epochs):
        # 动态调整学习率
        lr = get_lr(epoch, num_epochs, learning_rate, config.LR_WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Teacher Forcing 衰减
        tf_ratio = max(0.1, config.TEACHER_FORCING_RATIO * (1 - epoch / num_epochs))

        epoch_loss = 0
        num_batches = 0

        for batch_src, batch_tgt, batch_src_len, batch_tgt_len in train_loader:
            batch_src = batch_src.to(device)
            batch_tgt = batch_tgt.to(device)
            batch_src_len = batch_src_len.to(device)

            optimizer.zero_grad()
            outputs = model(batch_src, batch_src_len, batch_tgt, tf_ratio)

            output_dim = outputs.size(2)
            outputs_flat = outputs[:, 1:, :].contiguous().view(-1, output_dim)
            target_flat = batch_tgt[:, 1:].contiguous().view(-1)

            loss = criterion(outputs_flat, target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        # 每10轮打印
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | TF: {tf_ratio:.2f}")

            model.eval()
            with torch.no_grad():
                test_src = torch.LongTensor(np.array([source[0]])).to(device)
                test_len = torch.LongTensor(np.array([src_lengths[0]])).to(device)
                pred_ids = model.chat(test_src, test_len)[0].cpu().tolist()

                input_text = ''.join(id2word.get(i, '') for i in source[0]
                                     if i not in [config.PAD_ID, config.BOS_ID, config.EOS_ID])
                pred_text = ''.join(id2word.get(i, '') for i in pred_ids
                                    if i not in [config.PAD_ID, config.BOS_ID, config.EOS_ID])
                target_text = ''.join(id2word.get(i, '') for i in target[0]
                                      if i not in [config.PAD_ID, config.BOS_ID, config.EOS_ID])

                print(f"  输入: {input_text}")
                print(f"  期望: {target_text}")
                print(f"  预测: {pred_text}")
            model.train()

        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizer, epoch + 1, losses)

    save_checkpoint(model, optimizer, num_epochs, losses)
    plot_loss_curve(losses)
    return model, losses


def save_checkpoint(model, optimizer, epoch, losses):
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, config.MODEL_FILE)
    print(f"[训练] 模型已保存 (epoch {epoch})")


def plot_loss_curve(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='#2196F3', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Seq2Seq Chatbot Training Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(config.BASE_DIR, 'training_loss.png')
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"[训练] 损失曲线已保存到: {loss_path}")


def main():
    set_seed(42)
    source, target, src_lengths, tgt_lengths, vocab, word2id, id2word, embedding_matrix = load_data()
    train_loader = create_dataloader(source, target, src_lengths, tgt_lengths, config.BATCH_SIZE)

    vocab_size = len(vocab)
    model = Seq2SeqChatbot(vocab_size, embedding_matrix)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[训练] 模型参数量: {total_params:,}")

    model, losses = train_model(
        model, train_loader, config.NUM_EPOCHS, config.LEARNING_RATE,
        word2id, id2word, source, target, src_lengths, tgt_lengths, vocab_size
    )

    print()
    print(f"{'='*60}")
    print(f"训练完成！最终损失: {losses[-1]:.4f}")
    print(f"模型保存在: {config.MODEL_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
