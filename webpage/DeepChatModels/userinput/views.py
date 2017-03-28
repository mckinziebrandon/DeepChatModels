from flask import render_template, request
from userinput import app
from .forms import ChatForm

@app.route('/')
@app.route('/index', methods=['GET'])
def index():
    form = ChatForm()
    return render_template('index.html',
        form=form)

@app.route('/chat', methods=['POST'])
def chat():
    form = ChatForm(request.form)
    form.validate_on_submit()
    return form.userinput[::-1]