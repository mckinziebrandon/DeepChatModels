from datetime import datetime
import os
import yaml
import json

from flask import make_response, flash
from werkzeug.exceptions import HTTPException
from flask_admin.contrib import sqla

from . import main
from .. import db, web_bot, admin, basic_auth, api

from flask import redirect, current_app
from flask import render_template
from flask import session, url_for, request
from flask_cors import cross_origin
from flask_restful import Resource, fields

from .forms import ChatForm, UserForm
from ..models import User, Chatbot, Conversation, Turn
from .. import models
from pydoc import locate


@main.context_processor
def inject_enumerate():
    return dict(enumerate=enumerate)


@main.before_app_first_request
def load_gloabal_data():
    """Create the cornell_bot to be used for chat session."""
    session['start_time'] = None


@main.route('/')
@main.route('/index')
@cross_origin()
def index():
    # Create the (empty) forms that user can fill with info.
    user_form = UserForm()
    chat_form = ChatForm()
    return render_template('index.html',
                           user=session.get('user'),
                           user_form=user_form,
                           chat_form=chat_form)


@main.route('/about')
@cross_origin()
def about():
    return render_template('about.html', user=session.get('user', 'Anon'))


@main.route('/plots')
def plots():
    return render_template('plots.html',
                           user=session.get('user', 'Anon'))


def update_database(user_message, bot_response):
    """Fill database (db) with new input-response, and associated data."""

    # 1. Get the User db.Model.
    user = get_database_model('User', filter=session.get('user', 'Anon'))

    # 2. Get the Chatbot db.Model.
    bot_name = ChatAPI.bot_name
    chatbot = get_database_model('Chatbot', filter=bot_name)

    # 3. Get the Conversation db.Model.
    if session.get('start_time') is None:
        session['start_time'] = datetime.utcnow()
    conversation = get_database_model('Conversation',
                                      filter=session.get('start_time'),
                                      user=user,
                                      chatbot=chatbot)

    # 4. Get the Turn db.model. (called get adds it to the db if not there).
    _ = get_database_model('Turn',
                           user_message=user_message,
                           chatbot_message=bot_response,
                           conversation=conversation)
    db.session.commit()


def get_database_model(class_name, filter=None, **kwargs):
    model_class = getattr(models, class_name)
    assert model_class is not None, 'db_model for %s is None.' % class_name

    if filter is not None:
        if class_name == 'Conversation':
            filter_kw = {'start_time': filter}
        else:
            filter_kw = {'name': filter}
        db_model = model_class.query.filter_by(**filter_kw).first()
    else:
        db_model = None
        filter_kw = {}

    if db_model is None:
        db_model = model_class(**filter_kw, **kwargs)
        db.session.add(db_model)
    return db_model


# -------------------------------------------------------
# APIs
# -------------------------------------------------------

class UserAPI(Resource):

    def post(self):
        name = request.values.get('name', 'Anon')
        session['user'] = name
        user_model = get_database_model('User', filter=name)
        return {'name': user_model.name}


class ChatAPI(Resource):
    # Class attributes. This is convenient since we only want one active
    # bot at any given time.
    bot_name = 'Unk Bot'
    bot = None

    def __init__(self, data_name):
        if ChatAPI.bot_name != data_name:
            ChatAPI.bot_name = data_name
            ChatAPI.bot = web_bot.FrozenBot(frozen_model_dir=data_name,
                                            is_testing=current_app.testing)
            config = ChatAPI.bot.config
            _ = get_database_model('Chatbot',
                                   filter=ChatAPI.bot_name,
                                   dataset=config['dataset'],
                                   **config['model_params'])
            db.session.commit()
            # TODO: delete this after refactor rest of file.
            session['data_name'] = data_name

    def post(self):
        print('post received')
        print('request:', request)
        user_message = request.values.get('user_message')
        print('user_message = ', user_message)
        bot_response = self.bot(user_message)
        print('resp:', bot_response)
        update_database(user_message, bot_response)
        return {'response': bot_response,
                'bot_name': ChatAPI.bot_name}


class RedditAPI(ChatAPI):
    def __init__(self):
        super(RedditAPI, self).__init__('reddit')


class CornellAPI(ChatAPI):
    def __init__(self):
        super(CornellAPI, self).__init__('cornell')


class UbuntuAPI(ChatAPI):
    def __init__(self):
        super(UbuntuAPI, self).__init__('ubuntu')

api.add_resource(UserAPI, '/user/')
api.add_resource(RedditAPI, '/chat/reddit/')
api.add_resource(CornellAPI, '/chat/cornell/')
api.add_resource(UbuntuAPI, '/chat/ubuntu/')

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
