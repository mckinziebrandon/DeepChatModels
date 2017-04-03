from flask     import Flask
from flask_wtf import CSRFProtect
from utils import io_utils

# Don't judge me.
from pydoc import locate
from main import FLAGS
FLAGS.pretrained_dir = "pretrained/reddit"
config = io_utils.parse_config(FLAGS)
dataset = locate(config['dataset'])(config['dataset_params'])
bot = locate(config['model'])(dataset, config)

app = Flask(__name__)
app.config.from_object('config') # tells flask to read / use config.py
csrf = CSRFProtect(app)

from webpage.deepchat import views       # import at the end avoids circular imports
