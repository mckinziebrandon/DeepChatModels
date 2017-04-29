#!/usr/bin/env python3

"""run.py: Starts up the development web server with our application.

Command-line interface (pg 18 of flask reference):
    1. Launch the usual way (app.run(debug=True)):
    --> python3 run.py runserver

    2. Start a Python shell session in the context of the applciation.
    --> python3 run.py shell
"""

from deepchat import manager
manager.run()
