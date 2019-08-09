from flask_wtf import Form
from wtforms import TextAreaField
from wtforms.validators import Required


class ConsultaForm(Form):
    musica = TextAreaField(u'Musica', [
        Required(message='Por favor informe parte da musica a ser consultada')])
