from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
chat_bot = Table('chat_bot', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('dataset', String(length=64)),
    Column('base_cell', String(length=64)),
    Column('encoder', String(length=64)),
    Column('decoder', String(length=64)),
    Column('learning_rate', Float),
    Column('num_layers', Integer),
    Column('state_size', Integer),
)

conversation = Table('conversation', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('user_id', Integer),
    Column('chatbot_id', Integer),
)

turn = Table('turn', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('user_id', Integer),
    Column('chatbot_id', Integer),
    Column('user_message', Text),
    Column('chatbot_message', Text),
)

user = Table('user', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('name', String(length=64)),
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['chat_bot'].create()
    post_meta.tables['conversation'].create()
    post_meta.tables['turn'].create()
    post_meta.tables['user'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['chat_bot'].drop()
    post_meta.tables['conversation'].drop()
    post_meta.tables['turn'].drop()
    post_meta.tables['user'].drop()
