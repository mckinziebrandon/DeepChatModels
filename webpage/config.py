import os
basedir = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_TRACK_MODIFICATIONS = False  # Suppress error from package itself.
# Path of our db file. Required by Flask-SQLAlchemy extension.
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
# Folder where we'll store SQLAlchemy-migrate data files.
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')


class Config:
    #TESTING = True  # used internally by Flask instances.
    # Activates the cross-site request forgery prevention.
    WTF_CSRF_ENABLED = True
    # Used to create cryptographic token used to valide a chat_form.
    SECRET_KEY = 'you-might-guess-if-you-are-clever'

    # ---------- SQLAlchemy Configuration ----------
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Suppress error from package itself.
    # Path of our db file. Required by Flask-SQLAlchemy extension.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    # Folder where we'll store SQLAlchemy-migrate data files.
    SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
    # Commit the database when app closes.
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
