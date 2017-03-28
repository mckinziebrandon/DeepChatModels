from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

# Creates a form to wrap userinput to chatbot in.
class ChatForm(FlaskForm):
    userinput = StringField('userinput', validators = [DataRequired()])
