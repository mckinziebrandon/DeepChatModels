from flask     import Flask
from flask_wtf import CSRFProtect


app = Flask(__name__)
app.config.from_object('config') # tells flask to read / use config.py
csrf = CSRFProtect(app)

from chatbot import DynamicBot, ChatBot, SimpleBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData
from utils import io_utils
from pydoc import locate
import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_string("pretrained_dir", None, "path to pretrained model dir.")
flags.DEFINE_string("config", None, "path to config (.yml) file.")
flags.DEFINE_string("model", "{}", "Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
flags.DEFINE_string("model_params", "{}", "")
flags.DEFINE_string("dataset", "{}", "Options: data.{Cornell,Ubuntu,WMT}.")
flags.DEFINE_string("dataset_params", "{}", "")
FLAGS = flags.FLAGS

# Don't judge me.
FLAGS.pretrained_dir = "pretrained/reddit"
config = io_utils.parse_config(FLAGS)
dataset = locate(config['dataset'])(config['dataset_params'])
#bot = locate(config['model'])(dataset, config)
#sess = bot.sess
print("doneskies")


from webpage.deepchat import views       # import at the end avoids circular imports
