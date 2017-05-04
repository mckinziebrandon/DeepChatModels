"""apps/forms.py: """

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, \
    TextField, TextAreaField, HiddenField
from wtforms.validators import DataRequired, InputRequired
from wtforms.validators import ValidationError


def bad_chars(form, string_field):
    for c in r";'`":
        if c in string_field.data:
            raise ValidationError('DONT TYPE DAT')


class ChatForm(FlaskForm):
    """Creates a chat_form for users to enter input."""
    message = StringField('message', validators=[DataRequired()])
    submit = SubmitField('Submit')


class UserForm(FlaskForm):
    """Form for creating/editing a user."""
    name = StringField(label='name',
                       id='user-name',
        validators=[DataRequired(), bad_chars])
    submit = SubmitField(label='Submit')


class SentencePairForm(FlaskForm):
    input_sentence = StringField(
        label='input-sentence',
        id='input-sentence',
        validators=[DataRequired()])
    response_sentence = StringField(
        label='response-sentence',
        id='response-sentence',
        validators=[DataRequired()])
    submit = SubmitField(label='Submit')


