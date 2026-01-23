from flask import Flask
from . import main_view, wafer_view

def register_routes(app: Flask):
    app.register_blueprint(main_view.bp)
    app.register_blueprint(wafer_view.bp)