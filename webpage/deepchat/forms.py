from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired


class ChatForm(FlaskForm):
    """Creates a form for users to enter input."""
    message = StringField('message', validators=[DataRequired()])
