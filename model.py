# -*- coding: utf-8 -*-
"""
Seq2Seq + Attention 聊天机器人模型
架构：双层LSTM编码器 → Luong注意力 → 双层LSTM解码器
新增：Beam Search 解码策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Encoder(nn.Module):
    """编码器：双层LSTM，将输入序列编码为上下文向量"""

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers,
                 dropout, embedding_matrix):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=config.PAD_ID)
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            emb_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        safe_lengths = src_lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, safe_lengths, batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class LuongAttention(nn.Module):
    """Luong注意力机制"""

    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        score = torch.bmm(energy, decoder_hidden.unsqueeze(2))
        score = score.squeeze(2)
        attn_weights = F.softmax(score, dim=1)
        return attn_weights


class Decoder(nn.Module):
    """解码器：双层LSTM + 注意力机制 + 输出映射层"""

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers,
                 dropout, embedding_matrix):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=config.PAD_ID)
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = True

        self.attention = LuongAttention(hidden_size)

        self.lstm = nn.LSTM(
            emb_size + hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.projection = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input_token))

        decoder_h = hidden[0][-1]
        attn_weights = self.attention(decoder_h, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        lstm_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.lstm(lstm_input, hidden)

        prediction = self.projection(
            torch.cat([output.squeeze(1), context.squeeze(1)], dim=1)
        )

        return prediction, hidden, attn_weights

    def forward(self, encoder_outputs, hidden, target, teacher_forcing_ratio=0.5):
        batch_size = target.size(0)
        max_len = target.size(1)

        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(target.device)
        input_token = target[:, 0:1]

        # 统一的 batch 级别 teacher forcing 决策，提高可复现性
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        for t in range(1, max_len):
            prediction, hidden, _ = self.forward_step(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = prediction

            if use_teacher_forcing:
                input_token = target[:, t:t+1]
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)

        return outputs

    def inference(self, encoder_outputs, hidden, max_len=None):
        """贪婪解码（单条推理用）"""
        max_len = max_len or config.MAX_INFER_LEN
        batch_size = encoder_outputs.size(0)

        input_token = torch.full((batch_size, 1), config.BOS_ID,
                                  dtype=torch.long, device=encoder_outputs.device)

        decoded_ids = []
        for _ in range(max_len):
            prediction, hidden, _ = self.forward_step(input_token, hidden, encoder_outputs)
            input_token = prediction.argmax(dim=1, keepdim=True)
            decoded_ids.append(input_token)

            if (input_token == config.EOS_ID).all():
                break

        return torch.cat(decoded_ids, dim=1)


class BeamSearchDecoder:
    """Beam Search 解码器，比贪婪解码能找到更优的回复序列"""

    def __init__(self, decoder, beam_size=None):
        self.decoder = decoder
        self.beam_size = beam_size or config.BEAM_SIZE

    def decode(self, encoder_output, encoder_hidden):
        """
        encoder_output: (1, src_len, hidden)
        encoder_hidden: (h_n, c_n)
        返回: 最优序列的ID列表
        """
        device = encoder_output.device
        beam_size = self.beam_size

        # 简化实现：逐步扩展
        # 使用 (log_prob, sequence, hidden_state) 的列表
        h, c = encoder_hidden
        # 取第一个样本的 hidden state，保持层数维度 (num_layers, 1, hidden)
        beams = [(0.0, [config.BOS_ID], (h[:, :1, :].contiguous(), c[:, :1, :].contiguous()))]
        completed = []

        for _ in range(config.MAX_INFER_LEN):
            all_candidates = []
            for score, seq, (h_state, c_state) in beams:
                if seq[-1] == config.EOS_ID:
                    completed.append((score, seq))
                    continue

                input_tok = torch.LongTensor([[seq[-1]]]).to(device)
                pred, (new_h, new_c), _ = self.decoder.forward_step(
                    input_tok, (h_state, c_state), encoder_output
                )
                log_probs = F.log_softmax(pred, dim=-1).squeeze(0)
                topk_scores, topk_ids = log_probs.topk(beam_size)

                for i in range(beam_size):
                    new_score = score + topk_scores[i].item()
                    new_seq = seq + [topk_ids[i].item()]
                    all_candidates.append((new_score, new_seq, (new_h, new_c)))

            if not all_candidates:
                break

            # 按分数排序，保留 top beam_size
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]

        # 加入未完成的beam
        for score, seq, _ in beams:
            completed.append((score, seq))

        # 长度归一化后选最优
        if not completed:
            return [config.BOS_ID]

        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return best[1]


class Seq2SeqChatbot(nn.Module):
    """Seq2Seq聊天机器人完整模型"""

    def __init__(self, vocab_size, embedding_matrix):
        super(Seq2SeqChatbot, self).__init__()
        self.encoder = Encoder(
            vocab_size, config.EMB_SIZE, config.HIDDEN_SIZE,
            config.NUM_LAYERS, config.DROPOUT, embedding_matrix
        )
        self.decoder = Decoder(
            vocab_size, config.EMB_SIZE, config.HIDDEN_SIZE,
            config.NUM_LAYERS, config.DROPOUT, embedding_matrix
        )
        self.beam_decoder = BeamSearchDecoder(self.decoder)

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=None):
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = config.TEACHER_FORCING_RATIO

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        outputs = self.decoder(encoder_outputs, hidden, tgt, teacher_forcing_ratio)
        return outputs

    def chat(self, src, src_lengths):
        """推理对话（贪婪解码）"""
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        decoded_ids = self.decoder.inference(encoder_outputs, hidden)
        return decoded_ids

    def chat_beam(self, src, src_lengths):
        """推理对话（Beam Search 解码，质量更好）"""
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_lengths)
            best_ids = self.beam_decoder.decode(encoder_outputs, hidden)
        # 去掉开头的BOS
        if best_ids and best_ids[0] == config.BOS_ID:
            best_ids = best_ids[1:]
        return best_ids
