# -*- coding: utf-8 -*-
"""
聊天机器人Web应用
基于Flask的Web交互界面，支持思考日志显示
"""
import os
import time
from flask import Flask, render_template, request, jsonify
import config

app = Flask(__name__)

engine = None

def get_engine():
    global engine
    if engine is None:
        from chat import ChatbotEngine
        engine = ChatbotEngine()
        engine.load()
    return engine


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({"reply": "请输入一些内容再发送哦", "thinking": {}})
    try:
        bot = get_engine()
        result = bot.respond_detailed(user_msg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"reply": f"抱歉，处理出错了: {str(e)}", "thinking": {}})


@app.route('/api/stats')
def stats_api():
    try:
        bot = get_engine()
        return jsonify(bot.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "model": "Seq2Seq + Luong Attention"})


if __name__ == '__main__':
    print(f"[Web] 启动聊天机器人Web服务 http://{config.WEB_HOST}:{config.WEB_PORT}")
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False)
