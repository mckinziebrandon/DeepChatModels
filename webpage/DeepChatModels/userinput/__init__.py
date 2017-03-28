from flask import Flask

app = Flask(__name__)
app.config.from_object('config') # tells flask to read/use config.py

from userinput import views # Import at the end avoids circular import issues