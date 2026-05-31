# -*- coding: utf-8 -*-
"""
聊天机器人配置文件
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 路径配置 ====================
CORPUS_DIR = os.path.join(BASE_DIR, "corpus", "dialog")
IDS_DIR = os.path.join(BASE_DIR, "ids")
CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
TMP_DIR = os.path.join(BASE_DIR, "tmp")

IDS_FILE = os.path.join(IDS_DIR, "ids.json")
EMBEDDING_FILE = os.path.join(TMP_DIR, "embedding_matrix.npy")
WORD2VEC_FILE = os.path.join(TMP_DIR, "word2vec.model")
MODEL_FILE = os.path.join(CKPT_DIR, "chatbot_model.pt")
VOCAB_FILE = os.path.join(TMP_DIR, "vocab.json")

# ==================== 特殊标记 ====================
PAD_TOKEN = "_PAD"
BOS_TOKEN = "_BOS"
EOS_TOKEN = "_EOS"
UNK_TOKEN = "_UNK"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ==================== 模型超参数 ====================
EMB_SIZE = 128
HIDDEN_SIZE = 192
NUM_LAYERS = 1
DROPOUT = 0.1          # 适当正则化
LEARNING_RATE = 0.002
BATCH_SIZE = 128
NUM_EPOCHS = 30
TEACHER_FORCING_RATIO = 0.5
MAX_INFER_LEN = 30
CLIP_GRAD = 1.0
MIN_FREQ = 1

# ==================== 训练优化 ====================
LABEL_SMOOTHING = 0.1   # Label Smoothing，防止过度自信
LR_WARMUP_EPOCHS = 5    # 学习率预热轮次
WEIGHT_DECAY = 1e-5     # 权重衰减，防过拟合

# ==================== Beam Search ====================
BEAM_SIZE = 3           # Beam Search 宽度（1=贪婪解码）

# ==================== Web配置 ====================
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000