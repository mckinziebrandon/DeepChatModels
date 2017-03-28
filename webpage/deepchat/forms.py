from flask_wtf import FlaskForm
from wtforms   import StringField
from wtforms.validators import DataRequired

# Creates a form for users to enter input.
class ChatForm(FlaskForm):
    message = StringField('message', validators=[ DataRequired() ])
