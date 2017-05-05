from datetime import datetime
import yaml

from flask import make_response, flash
from werkzeug.exceptions import HTTPException
from flask_admin.contrib import sqla

from . import main
from .. import db, web_bot, admin, basic_auth

from flask import redirect, current_app
from flask import render_template
from flask import session, url_for, request
from flask_cors import cross_origin

from .forms import ChatForm, UserForm
from ..models import User, Chatbot, Conversation, Turn


@main.before_app_first_request
def load_gloabal_data():
    """Create the cornell_bot to be used for chat session."""
    global cornell_bot, reddit_bot

    session['start_time'] = None

    cornell_bot = web_bot.FrozenBot(frozen_model_dir='cornell',
                                    is_testing=current_app.testing)
    reddit_bot = web_bot.FrozenBot(frozen_model_dir='reddit',
                                   is_testing=current_app.testing)


@main.route('/', methods=['GET', 'POST'])
@main.route('/index', methods=['GET', 'POST'])
@cross_origin()
def index():

    # Create the (empty) forms that user can fill with info.
    user_form = UserForm()
    chat_form = ChatForm()

    if request.method == 'POST':
        if user_form.name.data:
            session['user'] = user_form.name.data
            # TODO: is this used anymore?
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


@main.route('/about/')
@main.route('/about')
@cross_origin()
def about():
    return render_template('about.html', user=session.get('user', 'Anon'))


@main.route('/chat', methods=['GET', 'POST'])
@cross_origin()
def chat():

    # Get the bot's response.
    user_message = session.get('user_message')
    chatbot_message = get_active_bot()(user_message)

    user = get_or_create_user(session.get('user', 'Anon'))
    if session.get('start_time') is None:
        session['start_time'] = datetime.utcnow()
    conversation = get_or_create_conversation(session.get('start_time'), user)
    turn = Turn(user_message=user_message,
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


def get_active_bot():
    if session.get('data_name') == 'cornell':
        return cornell_bot
    elif session.get('data_name') == 'reddit':
        return reddit_bot
    else:
        flash("Can't find the active bot!")


def get_or_create_conversation(time, user):

    conversation = Conversation.query.filter_by(start_time=time).first()
    if conversation is None:

        if current_app.testing:
            bot_name = 'Reverse TestBot'
        else:
            bot_name = 'Baby {}'.format(session.get('data_name', 'Unknown Bot'))

        # Get or create the bot (db.Model) . . .
        chatbot = Chatbot.query.filter_by(name=bot_name).first()
        if chatbot is None:
            chatbot = Chatbot(bot_name, get_active_bot().config)

        conversation = Conversation(start_time=time,
                                    user=user,
                                    chatbot=chatbot)
    return conversation

# -------------------------------------------------------
# ADMIN: Authentication for the admin (me) on /admin.
# -------------------------------------------------------


class AuthException(HTTPException):
    def __init__(self, message):
        super().__init__(message, make_response(
            "You could not be authenticated. Please refresh the page.", 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'}))


class ModelView(sqla.ModelView):
    def is_accessible(self):
        if not basic_auth.authenticate():
            raise AuthException('Not authenticated.')
        else:
            return True

    def inaccessible_callback(self, name, **kwargs):
        return redirect(basic_auth.challenge())

admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(Chatbot, db.session))
admin.add_view(ModelView(Conversation, db.session))
admin.add_view(ModelView(Turn, db.session))


