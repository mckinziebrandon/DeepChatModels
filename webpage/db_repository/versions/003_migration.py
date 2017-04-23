from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
chat_bot = Table('chat_bot', pre_meta,
    Column('id', INTEGER, primary_key=True, nullable=False),
    Column('dataset', VARCHAR(length=64)),
    Column('base_cell', VARCHAR(length=64)),
    Column('encoder', VARCHAR(length=64)),
    Column('decoder', VARCHAR(length=64)),
    Column('learning_rate', FLOAT),
    Column('num_layers', INTEGER),
    Column('state_size', INTEGER),
    Column('name', VARCHAR(length=64)),
)

chatbot = Table('chatbot', post_meta,
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


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['chat_bot'].drop()
    post_meta.tables['chatbot'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['chat_bot'].create()
    post_meta.tables['chatbot'].drop()
