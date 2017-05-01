"""deepchat/__init__.py: Initialize session objects."""
from flask import Flask
from flask_wtf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager, Shell
# from flask_moment import Moment
# from flask_pagedown import PageDown
from config import config

# Create flask application object.
app = Flask(__name__)
csrf = CSRFProtect(app)
# Read and use info in config.py.
app.config.from_object(config['default'])
app.config.update({'PREFERRED_URL_SCHEME': 'https'});
# Initialize our database.
db = SQLAlchemy(app)
# For better CLI.
manager = Manager(app)
# Nice thingy for displaying dates/times.
# moment = Moment(app)
# Client-sdie Markdown-to-HTML converter implemented in JS.
# pagedown = PageDown(app)


def make_shell_context():
    """Automatic imports when we want to play in the shell."""
    return dict(app=app)

manager.add_command('shell', Shell(make_context=make_shell_context))

from deepchat import views # import at the end avoids circular imports
