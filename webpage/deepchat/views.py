from flask import render_template
from flask import request

from deepchat import app
from deepchat import web_bot
from .forms import ChatForm


@app.before_first_request
def load_gloabal_data():
    """Create the bot to be used for chat session."""
    global bot
    # TODO: add support for querying frozen model about it's vocabulary.
    vocab_size = 40000
    bot = web_bot.FrozenBot(
        frozen_model_dir='reddit',
        vocab_size=vocab_size,
        is_testing=app.testing)


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    chat_form = ChatForm()
    return render_template('index.html', form=chat_form)


# TODO: why do we need the trailing slash here?
@app.route('/chat/', methods=['POST'])
def chat():
    chat_form = ChatForm()
    return bot(chat_form.message.data)
