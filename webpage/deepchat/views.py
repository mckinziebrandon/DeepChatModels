import pydoc
from datetime import datetime
from flask import render_template
from flask import request
from flask import session, url_for, request, g
from flask import flash, redirect

from deepchat import app, db
from deepchat import web_bot
from .forms import ChatForm, UserForm
from .models import User, Chatbot, Conversation, Turn
from wtforms import StringField


@app.before_first_request
def load_gloabal_data():
    """Create the cornell_bot to be used for chat session."""
    global cornell_bot, reddit_bot

    # TODO: add support for querying frozen model about it's vocabulary.
    vocab_size = 40000
    cornell_bot = web_bot.FrozenBot(
        frozen_model_dir='cornell',
        vocab_size=vocab_size,
        is_testing=app.testing)
    reddit_bot = web_bot.FrozenBot(
        frozen_model_dir='reddit',
        vocab_size=vocab_size,
        is_testing=app.testing)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():

    # Create the (empty) forms that user can fill with info.
    user_form = UserForm()
    chat_form = ChatForm()

    if request.method == 'POST':

        if user_form.name.data:
            session['user'] = user_form.name.data
            session['user_saved'] = True
        # Triggered by ajax call; simply return the response string.
        if chat_form.message.data:
            session['user_message'] = chat_form.message.data
            print('dataName:', request.form.get('dataName'))
            session['data_name'] = request.form.get('dataName')
            return redirect(url_for('.chat'), code=307)

    return render_template(
        'index.html',
        user=session.get('user'),
        user_form=user_form,
        chat_form=chat_form)


@app.route('/chat', methods=['POST'])
def chat():

    user_message = session.get('user_message')
    # Get the bot's response.
    if session.get('data_name') == 'cornell':
        chatbot_message = cornell_bot(user_message)
    else:
        chatbot_message = reddit_bot(user_message)

    user = get_or_create_user(session.get('user', 'Anonymous'))
    if session.get('start_time') is None:
        session['start_time'] = datetime.utcnow()
    conversation = get_or_create_conversation(session.get('start_time'), user)
    turn = Turn(
        user_message=user_message,
        chatbot_message=chatbot_message,
        conversation=conversation)
    db.session.add_all([turn, conversation])
    db.session.commit()
    return chatbot_message


def get_or_create_user(name):
    user = User.query.filter_by(name=name).first()
    if user is None:
        user = User(name=name)
    return user


def get_or_create_conversation(time, user):
    conversation = Conversation.query.filter_by(start_time=time).first()
    if conversation is None:
        chatbot = Chatbot(name='baby'+session.get('data_name'),
                          dataset=session.get('data_name'),
                          base_cell='GRUCell',
                          encoder='BasicEncoder',
                          decoder='BasicDecoder',
                          learning_rate=0.002,
                          num_layers=1,
                          state_size=512)
        conversation = Conversation(
            start_time=time,
            user=user,
            chatbot=chatbot)
    return conversation


@app.route('/about')
def about():
    return render_template('about.html', user=session.get('user'))