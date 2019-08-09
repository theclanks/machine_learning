# -*- coding: utf-8 -*-

from flask import Blueprint, request, render_template, \
                  flash, g, session, redirect, url_for
import functools


from app import clf

# Import module forms
from app.lyrics.forms import ConsultaForm

lyrics = Blueprint('lyrics', __name__, url_prefix='/lyrics')

# Set the route and accepted methods
@lyrics.route('/consulta/', methods=['GET', 'POST'])
def consulta():

    # If sign in form is submitted
    form = ConsultaForm(request.form)

    # Verify the sign in form
    if form.validate_on_submit():
        pred = clf.predict([form.musica.data])
        pred_prob = clf.predict_proba([form.musica.data])
        print(pred)
        print(pred_prob)
        form.resp = pred[0].capitalize().replace("_", " ")
        form.prob = map(functools.partial(round, ndigits=3), pred_prob[0])
        return render_template("lyrics/result.html", form=form)

    return render_template("lyrics/home.html", form=form)