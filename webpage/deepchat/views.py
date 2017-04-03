from flask import render_template
from flask import request

# IMPORT ALL THE THINGS.
from chatbot import DynamicBot, ChatBot, SimpleBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData
from utils import io_utils
from pydoc import locate
import tensorflow as tf

from webpage.deepchat import app, csrf, dataset, config
from .forms import ChatForm

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    chat_form = ChatForm()
    return render_template('index.html', form=chat_form)

@app.route('/chat/', methods=['POST'])
def chat():
    bot = locate(config['model'])(dataset, config)
    chat_form = ChatForm()
    return bot(chat_form.message.data)
