# -*- coding: utf-8 -*-
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
import joblib
# Import SQLAlchemy
#from flask.ext.sqlalchemy import SQLAlchemy

# Define the WSGI application object
app = Flask(__name__)
Bootstrap(app)

# Configurations
app.config.from_object('config')

# Classificador Lyrics
#pre_process = joblib.load("app/lyrics/Modelo/pre_process.pkl")
clf = joblib.load("app/lyrics/Modelo/md.pkl")

# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

# Import a module / component using its blueprint handler variable (mod_auth)
from app.lyrics.controllers import lyrics as lyrics_module

# Register blueprint(s)
app.register_blueprint(lyrics_module)

@app.route('/')
def hello():
    return redirect(url_for('lyrics.consulta'))
