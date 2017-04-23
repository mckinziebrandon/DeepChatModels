from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
turn = Table('turn', pre_meta,
    Column('id', INTEGER, primary_key=True, nullable=False),
    Column('user_id', INTEGER),
    Column('chatbot_id', INTEGER),
    Column('user_message', TEXT),
    Column('chatbot_message', TEXT),
)

turn = Table('turn', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('user_message', Text),
    Column('chatbot_message', Text),
    Column('conversation_id', Integer),
)

chat_bot = Table('chat_bot', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('name', String(length=64)),
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
    Column('start_time', DateTime),
    Column('user_id', Integer),
    Column('chatbot_id', Integer),
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['turn'].columns['chatbot_id'].drop()
    pre_meta.tables['turn'].columns['user_id'].drop()
    post_meta.tables['turn'].columns['conversation_id'].create()
    post_meta.tables['chat_bot'].columns['name'].create()
    post_meta.tables['conversation'].columns['start_time'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['turn'].columns['chatbot_id'].create()
    pre_meta.tables['turn'].columns['user_id'].create()
    post_meta.tables['turn'].columns['conversation_id'].drop()
    post_meta.tables['chat_bot'].columns['name'].drop()
    post_meta.tables['conversation'].columns['start_time'].drop()
