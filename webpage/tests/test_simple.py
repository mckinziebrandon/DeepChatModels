"""Unit tests for the application."""

import unittest
from flask import current_app
from deepchat import create_app, db


class TestSimple(unittest.TestCase):
    """Simple tests - ensuring the app can open/access database/etc."""

    def setUp(self):
        """Called before running a test."""
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        """Called after running a test."""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_app_exists(self):
        self.assertFalse(current_app is None)

    def test_app_is_testing(self):
        """Ensure we can access the right config specifications."""
        self.assertTrue(current_app.config['TESTING'])
