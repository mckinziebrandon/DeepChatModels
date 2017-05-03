"""Package constructor file for creating blueprint(s)."""

from flask import Blueprint
from flask_cors import CORS

# Blueprint(<blueprint name>, <module/package where blueprint is located>).
# Note: The <blueprint name> (main for us) defines the blueprint namespace.
main = Blueprint('main', __name__)
CORS(main, supports_credentials=True)

# By importing views and errors here, we cause the routes and error handlers
# to be associated with the blueprint.
from . import views, errors
