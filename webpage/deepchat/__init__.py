from flask     import Flask
from flask_wtf import CSRFProtect

from utils import io_utils
from utils import bot_freezer
from pydoc import locate

app = Flask(__name__)
app.config.from_object('config') # tells flask to read / use config.py
csrf = CSRFProtect(app)

# Get pretrained model configuration (dict) from path.
#config = io_utils.parse_config('pretrained/reddit')
#vocab_dicts = bot_freezer.get_frozen_vocab(config)
#dataset = locate(config['dataset'])(config['dataset_params'])

from webpage.deepchat import views       # import at the end avoids circular imports
