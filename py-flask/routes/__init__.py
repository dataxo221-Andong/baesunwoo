from flask import Flask
from . import main_view

def register_routes(app: Flask):
    app.register_blueprint(main_view.bp)