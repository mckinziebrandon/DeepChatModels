from flask     import Flask
from flask_wtf import CSRFProtect

app = Flask(__name__)
app.config.from_object('config') # tells flask to read / use config.py
csrf = CSRFProtect(app)

from deepchat import views       # import at the end avoids circular imports
