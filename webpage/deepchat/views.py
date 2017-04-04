from flask import render_template
from flask import request
from pydoc import locate

from webpage.deepchat import app, csrf, dataset, config
from .forms import ChatForm


@app.before_first_request
def load_gloabal_data():
    # StackOverflow: "Flask: Creating objects that remain over multiple requests"
    global bot
    bot = locate(config['model'])(dataset, config)


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    chat_form = ChatForm()
    return render_template('index.html', form=chat_form)


@app.route('/chat/', methods=['POST'])
def chat():
    chat_form = ChatForm()
    return bot(chat_form.message.data)
