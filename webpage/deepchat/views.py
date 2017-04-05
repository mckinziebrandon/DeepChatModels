from flask import render_template
from flask import request
from pydoc import locate

import os
from webpage.deepchat import app
from utils import bot_freezer
from .forms import ChatForm


@app.before_first_request
def load_gloabal_data():
    # StackOverflow: "Flask: Creating objects that remain over multiple requests"
    global bot
    data_name = 'reddit'
    vocab_size = 23765
    here = os.path.dirname(os.path.realpath(__file__))
    frozen_model_dir = os.path.join(here, 'static', 'assets', 'frozen_models', data_name)
    bot = bot_freezer.FrozenBot(frozen_model_dir=frozen_model_dir, vocab_size=vocab_size)


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    chat_form = ChatForm()
    return render_template('index.html', form=chat_form)


@app.route('/chat/', methods=['POST'])
def chat():
    chat_form = ChatForm()
    return bot(chat_form.message.data)
