# 聊天机器人 - NLP自然语言处理课程设计

基于 **检索+生成混合架构** 的中文聊天机器人，使用 PyTorch 实现。

## 项目概述

| 项目 | 说明 |
|------|------|
| 核心架构 | 检索匹配 + Seq2Seq生成 混合架构 |
| 生成模型 | LSTM Encoder + Luong Attention + LSTM Decoder |
| 解码策略 | Beam Search (k=3) + 质量过滤 |
| 分词工具 | jieba |
| 词向量 | Word2Vec (Gensim) |
| 深度学习框架 | PyTorch |
| Web框架 | Flask |
| 语料规模 | 98个文件，20000+问答对 |

## 项目结构

```
Natural_Language_robot/
├── config.py           # 全局配置（路径、超参数）
├── preprocess.py       # 语料预处理（分词-词典-ID向量-Word2Vec）
├── model.py            # Seq2Seq + Attention + Beam Search 模型定义
├── train.py            # 模型训练（Label Smoothing / 余弦退火 / 断点恢复）
├── chat.py             # 混合引擎（检索+生成）+ 质量过滤
├── web_app.py          # Flask Web API（支持思考日志）
├── main.py             # 主入口（统一命令调度）
├── requirements.txt    # Python依赖
├── corpus/dialog/      # 语料库文件夹 (98个.txt文件)
├── ids/                # 处理后的ID向量
├── ckpt/               # 模型checkpoint
├── tmp/                # Word2Vec模型、词向量矩阵、词典
└── templates/          # Web前端页面
    └── index.html      # 支持思考日志/深色模式/导出对话
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键运行（预处理 + 训练）

```bash
python main.py all
```

### 3. 分步运行

```bash
# 步骤1: 语料预处理
python main.py preprocess

# 步骤2: 模型训练
python main.py train

# 步骤3: 命令行对话
python main.py chat

# 或启动Web界面
python main.py web
```

## 混合架构设计

本项目采用 **检索+生成混合架构**，兼顾精准性和灵活性：

```
用户提问
   |
   +- jieba分词
   |
   +- 检索引擎（Jaccard相似度 + 关键词覆盖率）
   |     |
   |     +- 匹配度 >= 0.35 -> 直接返回语料库答案
   |     +- 匹配度 < 0.35  -> 进入生成模式
   |
   +- Seq2Seq生成（Beam Search k=3）
   |     |
   |     +- 生成质量好 -> 返回生成结果
   |     +- 生成质量差 -> 回退到检索结果
   |
   +- 质量过滤（去重复、去乱码）-> 最终回复
```

## 模型架构

```
输入: "你好，在吗"
       | jieba分词
Tokens: ["你", "好", "在", "吗"]
       | Word2Vec Embedding (128维)
+---------------------------+
|  Encoder (LSTM)           | -> encoder_outputs (192维)
+---------------------------+
       |
+---------------------------+
|  Luong Attention          | -> context vector
+---------------------------+
       |
+---------------------------+
|  Decoder (LSTM)           | -> 输出概率分布
+---------------------------+
       | Beam Search (k=3)
输出: "在的哦，请问有啥能帮你的么"
```

## 关键超参数 (config.py)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| EMB_SIZE | 128 | 词向量维度 |
| HIDDEN_SIZE | 192 | LSTM隐层大小 |
| NUM_LAYERS | 1 | LSTM层数 |
| LEARNING_RATE | 0.002 | 学习率 |
| BATCH_SIZE | 128 | 批大小 |
| NUM_EPOCHS | 30 | 训练轮次 |
| DROPOUT | 0.1 | Dropout比率 |
| LABEL_SMOOTHING | 0.1 | 标签平滑 |
| BEAM_SIZE | 3 | Beam Search宽度 |

## 语料库

- **规模**: 98个.txt文件，20000+问答对
- **格式**: UTF-8编码，奇数行为问题，偶数行为回答
- **覆盖主题**: 日常对话、学习、工作、情感、科技、旅游、美食、运动、心理、文化等20+个领域

```
你好，在吗
在的哦，请问有啥能帮你的么
你叫什么名字
我叫小智，是你的智能聊天助手
```

## Web界面功能

运行 `python main.py web` 启动 Web 服务，访问 http://127.0.0.1:5000

| 功能 | 说明 |
|------|------|
| 思考日志 | 实时显示分词、词ID、解码方法、检索分数 |
| 深色模式 | 一键切换暗色主题 |
| 导出对话 | 将聊天记录导出为txt文件 |
| 复制回复 | 一键复制机器人回复 |
| 快捷提问 | 6个预设问题按钮 |
| 模型信息 | 查看架构、参数量、检索语料数等 |
| 搜索对话 | Ctrl+K 搜索历史消息 |
| 历史问题 | 查看过往提问记录 |

## 训练优化策略

| 策略 | 说明 |
|------|------|
| Label Smoothing | 防止模型过度自信，提升泛化 |
| 余弦退火学习率 | 预热+余弦衰减，收敛更细 |
| 梯度裁剪 | 防止梯度爆炸 |
| 权重衰减 | AdamW正则化，防过拟合 |
| Teacher Forcing衰减 | batch级统一决策，提高可复现性 |
| Word2Vec缓存校验 | 词表变化时自动重训 Word2Vec |
| 断点自动恢复 | 训练中断后可从上次断点继续 |

## 当前训练状态

- 已完成轮次: 30
- 当前Loss: 1.46
- 模型大小: 96.3 MB

## 扩展思路

- 增加更多高质量语料提升回复质量
- 尝试 Transformer / BERT 等模型对比效果
- 引入对话历史上下文
- 部署到服务器供多人使用
- 添加用户反馈机制，持续优化回复
- 升级检索系统为 BM25 或语义检索

## 许可

本项目为 NLP 自然语言处理课程设计作业。
