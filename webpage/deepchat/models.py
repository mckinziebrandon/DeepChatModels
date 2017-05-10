"""app/models.py: Tutorial IV - Databases.

database models: collection of classes whose purpose is to represent the
                 data that we will store in our database.

The ORM layer (SQLAlchemy) will do the translations required to map
objects created from these classes into rows in the proper database table.
    - ORM: Object Relational Mapper; links b/w tables corresp. to objects.
"""

from deepchat import db
import json


class User(db.Model):
    """A model that represents our users.

    Jargon/Parameters:
        - primary key: unique id given to each user.
        - varchar: a string.
        - db.Column parameter info:
            - index=True: allows for faster queries by associating a given column
                          with its own index. Use for values frequently looked up.
            - unique=True: don't allow duplicate values in this column.

    Fields:
        id: (db.Integer) primary_key for identifying a user in the table.
        name: (str)
        posts: (db.relationship)

    """

    # Fields are defined as class variables, but are used by super() in init.
    # Pass boolean args to indicate which fields are unique/indexed.
    # Note: 'unique' here means [a given user] 'has only one'.
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    # Relationships are not actual database fields (not shown on a db diagram).
    # - backref: *defines* a field that will be added to the instances of
    #             Posts that point back to this user.
    # - lazy='dynamic': "Instead of loading the items, return another query
    #                    object which we can refine before loading items.
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic')

    def __repr__(self):
        return "<User {0}>".format(self.name)


class Chatbot(db.Model):
    """Chatbot. Fields are the same as from yaml config files."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)  # TODO: make unique?
    dataset = db.Column(db.String(64))
    base_cell = db.Column(db.String(64))
    encoder = db.Column(db.String(64))
    decoder = db.Column(db.String(64))
    learning_rate = db.Column(db.Float)
    num_layers = db.Column(db.Integer)
    state_size = db.Column(db.Integer)
    conversations = db.relationship('Conversation', backref='chatbot', lazy='dynamic')

    def __init__(self, name, **bot_kwargs):
        self.name = (name or 'Unknown Bot')
        self.dataset = bot_kwargs['dataset']
        self.base_cell = bot_kwargs['base_cell']
        self.encoder = bot_kwargs['encoder']
        self.decoder = bot_kwargs['decoder']
        self.learning_rate = bot_kwargs['learning_rate']
        self.num_layers = bot_kwargs['num_layers']
        self.state_size = bot_kwargs['state_size']


    def __repr__(self):
        return json.dumps("<Chatbot {0}>".format(self.name))


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, index=True, unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    chatbot_id = db.Column(db.Integer, db.ForeignKey('chatbot.id'))
    turns = db.relationship('Turn', backref='conversation', lazy='dynamic')

    def __repr__(self):
        return '<Conversation between {0} and {1}>'.format(self.user_id, self.chatbot_id)


class Turn(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.Text)
    chatbot_message = db.Column(db.Text)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'))

    def __repr__(self):
        return 'User: {0}\nChatBot: {1}'.format(
            self.user_message, self.chatbot_message)
