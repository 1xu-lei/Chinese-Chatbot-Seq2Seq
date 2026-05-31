# -*- coding: utf-8 -*-
"""
语料库预处理模块
功能：读取语料库 → jieba分词 → 构建词典 → ID向量化 → Word2Vec训练 → 保存文件
"""
import os
import json
import numpy as np
import jieba
from gensim.models import Word2Vec
import config


def read_corpus(corpus_dir=None):
    """读取语料库文件，返回(问题列表, 回答列表)"""
    corpus_dir = corpus_dir or config.CORPUS_DIR
    corpus_files = os.listdir(corpus_dir)
    questions, answers = [], []

    for fname in corpus_files:
        fpath = os.path.join(corpus_dir, fname)
        if not os.path.isfile(fpath) or not fname.endswith('.txt'):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        for i in range(0, len(lines) - 1, 2):
            questions.append(lines[i])
            answers.append(lines[i + 1])

    print(f"[预处理] 读取到 {len(questions)} 对问答语料")
    return questions, answers


def tokenize(text):
    """使用jieba进行中文分词，过滤空白token"""
    return [w.strip() for w in jieba.cut(text) if w.strip()]


def build_vocab(questions, answers, min_freq=None):
    """构建词典，返回 (词典列表, word2id字典, id2word字典)"""
    min_freq = min_freq or config.MIN_FREQ
    freq = {}
    for q, a in zip(questions, answers):
        for word in tokenize(q) + tokenize(a):
            freq[word] = freq.get(word, 0) + 1

    # 特殊标记在前
    vocab = list(config.SPECIAL_TOKENS)
    for word, count in sorted(freq.items(), key=lambda x: -x[1]):
        if count >= min_freq and word not in vocab:
            vocab.append(word)

    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}
    print(f"[预处理] 词典大小: {len(vocab)} (含 {len(config.SPECIAL_TOKENS)} 个特殊标记)")
    return vocab, word2id, id2word


def text_to_ids(text, word2id):
    """将文本转为ID序列"""
    tokens = tokenize(text)
    ids = [word2id.get(w, config.UNK_ID) for w in tokens]
    return ids


def encode_pairs(questions, answers, word2id):
    """将问答对编码为ID序列，添加BOS/EOS标记"""
    source_ids, target_ids = [], []
    src_lengths, tgt_lengths = [], []

    for q, a in zip(questions, answers):
        src = text_to_ids(q, word2id)
        tgt = [config.BOS_ID] + text_to_ids(a, word2id) + [config.EOS_ID]
        # 过滤空序列
        if len(src) == 0 or len(tgt) <= 2:
            continue
        source_ids.append(src)
        target_ids.append(tgt)
        src_lengths.append(len(src))
        tgt_lengths.append(len(tgt))

    print(f"[预处理] 有效问答对: {len(source_ids)}")
    return source_ids, target_ids, src_lengths, tgt_lengths


def pad_sequences(sequences, max_len=None, pad_value=None):
    """将ID序列填充到相同长度"""
    pad_value = pad_value if pad_value is not None else config.PAD_ID
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    padded = []
    for s in sequences:
        padded.append(s + [pad_value] * (max_len - len(s)))
    return padded


def train_word2vec(source_ids, target_ids, emb_size=None):
    """训练Word2Vec词向量模型"""
    emb_size = emb_size or config.EMB_SIZE
    sentences = [[str(x) for x in ids] for ids in source_ids + target_ids]

    # 计算当前语料的词汇集，用于缓存一致性校验
    current_vocab = set()
    for sent in sentences:
        current_vocab.update(sent)
    current_vocab_size = len(current_vocab)

    if os.path.exists(config.WORD2VEC_FILE):
        model = Word2Vec.load(config.WORD2VEC_FILE)
        cached_vocab_size = len(model.wv)
        if cached_vocab_size < current_vocab_size * 0.9:
            print(f"[预处理] Word2Vec缓存词表过小({cached_vocab_size} vs {current_vocab_size})，重新训练")
            os.remove(config.WORD2VEC_FILE)
            model = Word2Vec(
                sentences, vector_size=emb_size, window=5,
                min_count=1, workers=1, sg=1, epochs=15
            )
            model.save(config.WORD2VEC_FILE)
            print("[预处理] Word2Vec模型重新训练并保存完毕")
        else:
            print(f"[预处理] Word2Vec模型已存在且词表一致({cached_vocab_size})，直接加载")
    else:
        print("[预处理] 开始训练Word2Vec词向量模型...")
        model = Word2Vec(
            sentences, vector_size=emb_size, window=5,
            min_count=1, workers=1, sg=1, epochs=15
        )
        model.save(config.WORD2VEC_FILE)
        print("[预处理] Word2Vec模型训练并保存完毕")
    return model




def build_embedding_matrix(w2v_model, vocab, word2id, emb_size=None):
    """构建词向量矩阵"""
    emb_size = emb_size or config.EMB_SIZE
    vocab_size = len(vocab)
    matrix = np.zeros((vocab_size, emb_size))

    np.random.seed(42)
    for word, idx in word2id.items():
        key = str(idx)
        if key in w2v_model.wv:
            matrix[idx] = w2v_model.wv[key]
        else:
            matrix[idx] = np.random.normal(0, 0.1, emb_size)

    print(f"[预处理] 词向量矩阵构建完成: {matrix.shape}")
    return matrix


def run_preprocess():
    """执行完整预处理流程"""
    os.makedirs(config.IDS_DIR, exist_ok=True)
    os.makedirs(config.TMP_DIR, exist_ok=True)

    # 如果预处理结果已存在，跳过
    if (os.path.exists(config.IDS_FILE)
            and os.path.exists(config.EMBEDDING_FILE)
            and os.path.exists(config.VOCAB_FILE)):
        print("[预处理] 预处理结果已存在，跳过（如需重新处理请删除 ids/ 和 tmp/ 目录）")
        return None, None

    # 1. 读取语料
    questions, answers = read_corpus()

    # 2. 构建词典
    vocab, word2id, id2word = build_vocab(questions, answers)

    # 3. 编码为ID序列
    source_ids, target_ids, src_lengths, tgt_lengths = encode_pairs(
        questions, answers, word2id
    )

    # 4. 填充序列
    source_padded = pad_sequences(source_ids)
    target_padded = pad_sequences(target_ids)

    # 5. 训练Word2Vec
    w2v_model = train_word2vec(source_ids, target_ids)

    # 6. 构建词向量矩阵
    embedding_matrix = build_embedding_matrix(w2v_model, vocab, word2id)

    # 7. 保存处理后的数据
    ids_data = {
        "source": source_padded,
        "target": target_padded,
        "source_lengths": src_lengths,
        "target_lengths": tgt_lengths,
        "vocab": vocab,
    }
    with open(config.IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(ids_data, f, ensure_ascii=False)
    print(f"[预处理] ID数据已保存到: {config.IDS_FILE}")

    np.save(config.EMBEDDING_FILE, embedding_matrix)
    print(f"[预处理] 词向量矩阵已保存到: {config.EMBEDDING_FILE}")

    # 8. 保存词典映射
    vocab_data = {"word2id": word2id, "id2word": {str(k): v for k, v in id2word.items()}}
    with open(config.VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False)
    print(f"[预处理] 词典映射已保存到: {config.VOCAB_FILE}")

    # 统计信息
    print(f"\n{'='*50}")
    print(f"语料对数: {len(source_ids)}")
    print(f"词典大小: {len(vocab)}")
    print(f"词向量维度: {config.EMB_SIZE}")
    print(f"Source最大长度: {max(src_lengths)}")
    print(f"Target最大长度: {max(tgt_lengths)}")
    print(f"{'='*50}")

    return ids_data, embedding_matrix


if __name__ == "__main__":
    run_preprocess()