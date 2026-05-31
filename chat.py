# -*- coding: utf-8 -*-
"""
聊天机器人测试模块
混合检索+生成 架构：先在语料库找答案，找不到再用模型生成
"""
import os
import json
import time
import torch
import jieba
import numpy as np
from collections import Counter

import config
from model import Seq2SeqChatbot


class RetrievalIndex:
    """语料库检索引擎，基于关键词匹配"""

    def __init__(self):
        self.qa_pairs = []
        self.loaded = False

    def load(self, corpus_dir):
        """加载语料库到内存"""
        seen = set()
        for fname in os.listdir(corpus_dir):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(corpus_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for i in range(0, len(lines) - 1, 2):
                q, a = lines[i], lines[i+1]
                if q in seen or len(a) < 2:
                    continue
                seen.add(q)
                tokens = set(jieba.cut(q))
                tokens.discard(" ")
                self.qa_pairs.append((q, a, tokens))
        self.loaded = True
        print(f"[检索] 加载 {len(self.qa_pairs)} 对语料到索引")

    def search(self, query_tokens, top_k=3):
        """检索最相似的问答对"""
        if not self.qa_pairs:
            return []
        query_set = set(query_tokens)
        query_set.discard(" ")
        if not query_set:
            return []

        scored = []
        for q, a, t_set in self.qa_pairs:
            intersection = query_set & t_set
            union = query_set | t_set
            if not union:
                continue
            jaccard = len(intersection) / len(union)
            coverage = len(intersection) / len(query_set) if query_set else 0
            score = 0.3 * jaccard + 0.7 * coverage
            if score > 0.3:
                scored.append((score, q, a))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


class ChatbotEngine:
    """混合聊天机器人引擎：检索 + 生成"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.word2id = None
        self.id2word = None
        self.vocab_size = 0
        self.chat_count = 0
        self.retrieval = RetrievalIndex()
        self.RETRIEVAL_THRESHOLD = 0.35
        self.LOW_CONFIDENCE_THRESHOLD = 0.10

    def load(self):
        with open(config.VOCAB_FILE, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.word2id = vocab_data['word2id']
        self.id2word = {int(k): v for k, v in vocab_data['id2word'].items()}
        self.vocab_size = len(self.word2id)

        embedding_matrix = np.load(config.EMBEDDING_FILE)
        self.model = Seq2SeqChatbot(self.vocab_size, embedding_matrix)

        checkpoint = torch.load(config.MODEL_FILE, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.retrieval.load(config.CORPUS_DIR)

        mode = f"Beam Search (beam={config.BEAM_SIZE})" if config.BEAM_SIZE > 1 else "Greedy"
        print(f"[对话] 引擎加载完成 (词典: {self.vocab_size}, 设备: {self.device}, 解码: {mode})")

    def tokenize(self, text):
        return [w.strip() for w in jieba.cut(text) if w.strip()]

    def preprocess(self, text):
        tokens = self.tokenize(text)
        if not tokens:
            return None, None, tokens
        ids = [self.word2id.get(w, config.UNK_ID) for w in tokens]
        src = torch.LongTensor([ids]).to(self.device)
        src_len = torch.LongTensor([len(ids)]).to(self.device)
        return src, src_len, tokens

    def postprocess(self, pred_ids):
        words = []
        for idx in pred_ids:
            if idx in [config.PAD_ID, config.BOS_ID, config.EOS_ID]:
                continue
            word = self.id2word.get(idx, '')
            if word:
                words.append(word)
        return ''.join(words)

    def get_token_info(self, tokens):
        info = []
        for t in tokens:
            tid = self.word2id.get(t, config.UNK_ID)
            is_unk = (tid == config.UNK_ID)
            info.append({"word": t, "id": tid, "is_unk": is_unk})
        return info

    def _filter_response(self, text):
        """质量过滤：去除低质量回复"""
        if not text or len(text.strip()) < 2:
            return None
        for c in set(text):
            if text.count(c) > 3 and c not in "。，！？、":
                if c not in "的了是不我你他":
                    return None
        words = list(jieba.cut(text))
        if len(words) >= 4:
            for i in range(len(words) - 3):
                gram = "".join(words[i:i+3])
                rest = "".join(words[i+3:])
                if gram in rest:
                    return None
        return text.strip()

    def respond(self, user_input):
        result = self.respond_detailed(user_input)
        return result["reply"]

    def respond_detailed(self, user_input):
        self.chat_count += 1
        start_time = time.time()

        src, src_len, tokens = self.preprocess(user_input)
        if src is None:
            return {
                "reply": "你说的话我没听清，能再说一遍吗？",
                "thinking": {"tokens": [], "token_ids": [], "unknown_words": [],
                             "inference_time": 0, "method": "none",
                             "input_length": 0, "output_length": 0,
                             "retrieval_score": 0, "retrieval_source": "none"}
            }

        token_info = self.get_token_info(tokens)
        token_ids = [item["id"] for item in token_info]
        unknown_words = [item["word"] for item in token_info if item["is_unk"]]

        retrieval_results = self.retrieval.search(tokens)
        retrieval_score = retrieval_results[0][0] if retrieval_results else 0
        retrieval_source = "none"

        response = None
        method = ""
        pred_ids = []

        if retrieval_score >= self.RETRIEVAL_THRESHOLD:
            response = retrieval_results[0][2]
            method = f"检索匹配 (score={retrieval_score:.2f})"
            retrieval_source = "retrieval_high"
        else:
            if config.BEAM_SIZE > 1:
                pred_ids = self.model.chat_beam(src, src_len)
                method = f"Beam Search (k={config.BEAM_SIZE})"
            else:
                pred_ids = self.model.chat(src, src_len)[0].cpu().tolist()
                method = "Greedy Decode"

            response = self.postprocess(pred_ids)

            filtered = self._filter_response(response)
            if filtered:
                response = filtered
                retrieval_source = "generation"
            elif retrieval_results and retrieval_score > 0.3:
                response = retrieval_results[0][2]
                method += " → 检索回退 (score={:.2f})".format(retrieval_score)
                retrieval_source = "retrieval_fallback"
            else:
                retrieval_source = "generation"

        if not response or not response.strip():
            response = "抱歉，我还不太理解这个问题，能换个说法吗？"
            retrieval_source = "fallback"

        inference_time = time.time() - start_time

        reply_ids = [idx for idx in pred_ids if idx not in [config.PAD_ID, config.BOS_ID, config.EOS_ID]]
        reply_tokens = [self.id2word.get(idx, "?") for idx in reply_ids]

        thinking = {
            "tokens": tokens,
            "token_ids": token_ids,
            "unknown_words": unknown_words,
            "reply_tokens": reply_tokens,
            "reply_ids": reply_ids,
            "inference_time": round(inference_time * 1000, 1),
            "method": method,
            "input_length": len(tokens),
            "output_length": len(reply_tokens),
            "unknown_count": len(unknown_words),
            "vocab_size": self.vocab_size,
            "chat_count": self.chat_count,
            "retrieval_score": round(retrieval_score, 3),
            "retrieval_source": retrieval_source,
            "retrieval_candidates": [(round(s, 2), q, a[:30]) for s, q, a in retrieval_results[:3]]
        }

        return {"reply": response, "thinking": thinking}

    def get_stats(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "vocab_size": self.vocab_size,
            "total_params": total_params,
            "trainable_params": trainable,
            "device": str(self.device),
            "beam_size": config.BEAM_SIZE,
            "hidden_size": config.HIDDEN_SIZE,
            "emb_size": config.EMB_SIZE,
            "num_layers": config.NUM_LAYERS,
            "chat_count": self.chat_count,
            "retrieval_size": len(self.retrieval.qa_pairs)
        }


def main():
    engine = ChatbotEngine()
    engine.load()

    print("\n" + "="*50)
    print("  聊天机器人 - 检索+生成混合架构")
    print("  输入 bye 退出对话")
    print("="*50 + "\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'bye':
            print("机器人: 再见，期待下次和你聊天！")
            break

        result = engine.respond_detailed(user_input)
        t = result['thinking']
        print(f"机器人: {result['reply']}")
        print(f"  [分词] {t['tokens']}")
        print(f"  [方法] {t['method']} | 耗时 {t['inference_time']}ms")
        print(f"  [检索] score={t['retrieval_score']} source={t['retrieval_source']}")
        if t["unknown_words"]:
            print(f"  [未知词] {t['unknown_words']}")
        print()


if __name__ == "__main__":
    main()
