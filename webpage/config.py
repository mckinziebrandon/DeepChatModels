import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:

    DEFAULT_THEME = 'lumen'
    # Boolean: True if you == Brandon McKinze; False otherwise :)
    FLASK_PRACTICE_ADMIN = os.getenv('true')
    # Activates the cross-site request forgery prevention.
    WTF_CSRF_ENABLED = True
    # Used to create cryptographic token used to valide a form.
    SECRET_KEY = os.getenv('SECRET_KEY', 'not-really-a-secret-now')

    # SQLAlchemy configuration.
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True

    # Username/password for flask admin access.
    # COVER YOUR EYES - LOOK AWAY - NOTHING TO SEE HERE
    BASIC_AUTH_USERNAME = os.getenv('BASIC_AUTH_USERNAME', 'admin')
    BASIC_AUTH_PASSWORD = os.getenv('BASIC_AUTH_PASSWORD', 'password')

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    # Path of our db file. Required by Flask-SQLAlchemy extension.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data_dev.db')


class TestingConfig(Config):
    TESTING = True
    # Path of our db file. Required by Flask-SQLAlchemy extension.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data_test.db')


class ProductionConfig(Config):
    # Path of our db file. Required by Flask-SQLAlchemy extension.
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data.db')
    SESSION_COOKIE_HTTPONLY = False
    PREFERRED_URL_SCHEME = 'https'

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

