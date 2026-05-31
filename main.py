# -*- coding: utf-8 -*-
"""
聊天机器人主入口
用法:
    python main.py preprocess  - 执行语料预处理
    python main.py train       - 训练模型
    python main.py chat        - 命令行对话
    python main.py web         - 启动Web服务
    python main.py all         - 一键执行全部流程
"""
import sys
import os


def step_preprocess():
    """步骤1: 语料预处理"""
    print("\n" + "="*60)
    print("  步骤 1/3: 语料预处理")
    print("="*60)
    from preprocess import run_preprocess
    run_preprocess()


def step_train():
    """步骤2: 模型训练"""
    print("\n" + "="*60)
    print("  步骤 2/3: 模型训练")
    print("="*60)
    from train import main as train_main
    train_main()


def step_chat():
    """步骤3: 命令行对话"""
    print("\n" + "="*60)
    print("  步骤 3: 开始对话")
    print("="*60)
    from chat import main as chat_main
    chat_main()


def step_web():
    """启动Web服务"""
    print("\n" + "="*60)
    print("  启动Web聊天界面")
    print("="*60)
    import config
    print(f"  访问地址: http://{config.WEB_HOST}:{config.WEB_PORT}")
    print("="*60)
    from web_app import app
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False)


def run_all():
    """一键执行全部流程"""
    step_preprocess()
    step_train()
    print("\n" + "="*60)
    print("  全部流程执行完毕！")
    print("  运行 'python main.py chat' 进行命令行对话")
    print("  运行 'python main.py web' 启动Web聊天界面")
    print("="*60)


COMMANDS = {
    'preprocess': step_preprocess,
    'train': step_train,
    'chat': step_chat,
    'web': step_web,
    'all': run_all,
}


def print_usage():
    print("""
╔══════════════════════════════════════════════════╗
║        聊天机器人 - Seq2Seq + Attention           ║
║          NLP 自然语言处理课程设计                   ║
╠══════════════════════════════════════════════════╣
║  用法: python main.py <command>                   ║
║                                                  ║
║  可用命令:                                        ║
║    preprocess  - 语料预处理 (分词/词典/词向量)      ║
║    train       - 训练Seq2Seq模型                  ║
║    chat        - 命令行对话测试                    ║
║    web         - 启动Web聊天界面                   ║
║    all         - 一键执行 预处理→训练              ║
╚══════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print_usage()
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
