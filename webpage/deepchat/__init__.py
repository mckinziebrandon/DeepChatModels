"""deepchat/__init__.py: Initialize session objects."""

import os
from flask import Flask
from flask_wtf import CSRFProtect
from flask_moment import Moment
from flask_restful import Resource, Api
from flask_basicauth import BasicAuth
from flask_pagedown import PageDown
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from config import config

csrf = CSRFProtect()
# Initialize our database.
db = SQLAlchemy()
# Nice thingy for displaying dates/times.
moment = Moment()
# Client-sdie Markdown-to-HTML converter implemented in JS.
pagedown = PageDown()
# Flask-restful api interface.
api = Api()
# Database visualizer.
#name=os.getenv('APPENGINE_CONFIG', 'Development').title(),
admin = Admin(template_mode='bootstrap3')
# Basic authentication (mainly for using flask-admin).
basic_auth = BasicAuth()


def create_app(config_name):
    """The application factory, which allows the app to be created at runtime. 
    This is in contrast to before, where it was created in the global scope 
    (i.e. no way to apply configuration changes dynamically).
    
    Returns:
        app: the created application instance. Note that the app is still 
        missing routes and custom error page handlers, which will be handled 
        by blueprints.
    """

    from .main import main as main_blueprint

    # Create flask application object, and
    # read/use info in config.py.
    app = Flask(__name__)
    csrf.init_app(app)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Read and use info in config.py.
    app.config.update({'PREFERRED_URL_SCHEME': 'https'});

    # Initialize our database.
    db.init_app(app)
    # Nice thingy for displaying dates/times.
    moment.init_app(app)
    # Client-sdie Markdown-to-HTML converter implemented in JS.
    pagedown.init_app(app)
    #
    admin.name = config_name.title()
    admin.init_app(app)

    basic_auth.init_app(app)

    api.init_app(app)
    app.register_blueprint(main_blueprint)

    return app



