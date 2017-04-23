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
    """Create the bot to be used for chat session."""
    global bot, chatbot_model

    session['user'] = None
    session['user_saved'] = False

    # TODO: Just load the yaml config so we don't need to hardcode.
    chatbot_model = Chatbot(
        name='babycornell',
        dataset='cornell',
        base_cell='GRUCell',
        encoder='BasicEncoder',
        decoder='BasicDecoder',
        learning_rate=0.002,
        num_layers=1,
        state_size=512)

    # TODO: add support for querying frozen model about it's vocabulary.
    vocab_size = 40000
    bot = web_bot.FrozenBot(
        frozen_model_dir='cornell',
        vocab_size=vocab_size,
        is_testing=app.testing)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():

    # Create the (empty) forms that user can fill with info.
    user_form = UserForm()
    chat_form = ChatForm()

    if request.method == 'POST':
        if user_form.validate_on_submit() and user_form.submit.data:
            session['user'] = user_form.name.data
            session['user_saved'] = True

        # Triggered by ajax call; simply return the response string.
        if chat_form.message.data:
            # Send to chat endpoint; preserve POST HTTP code.
            session['user_message'] = chat_form.message.data
            return redirect(url_for('chat'), code=307)

    return render_template(
        'index.html',
        user=session.get('user'),
        user_form=user_form,
        chat_form=chat_form)


@app.route('/chat', methods=['POST'])
def chat():

    user_message = session.get('user_message')
    # Get the bot's response.
    chatbot_message = bot(user_message)

    if session.get('user'):
        user = get_or_create_user(session.get('user'))
    else:
        user = get_or_create_user('Anonymous')

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
        conversation = Conversation(
            start_time=time,
            user=user,
            chatbot=chatbot_model)
    return conversation


@app.route('/about')
def about():
    return render_template('about.html', user=session.get('user'))