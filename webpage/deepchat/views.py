from flask import render_template
from flask import request

from webpage.deepchat import app, csrf
from .forms import ChatForm


@app.route('/', methods=[ 'GET' ])
@app.route('/index', methods=[ 'GET' ])
def index():
    chat_form = ChatForm()
    return render_template('index.html', form=chat_form)

@app.route('/chat/', methods=[ 'POST' ])
def chat():
    chat_form = ChatForm()
    return chat_form.message.data[::-1]
