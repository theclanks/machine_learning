#!/bin/bash

echo "Instalando ambiente virtual"
pip install virtualenv

echo "Criando ambiente virtual"
virtualenv env

echo "Carregando e instalando bibliotecas"
source env/bin/activate

echo "Instalando ..."
pip install -r requirements-to-freeze.txt

echo "Instalando bibliotecas NLTK ... averaged_perceptron_tagger - floresta - mac_morpho - machado - punkt - stopwords - wordnet - words"
python -m nltk.downloader  averaged_perceptron_tagger floresta mac_morpho machado punkt stopwords wordnet words

#echo "Criando kernel"
#ipython kernel install --user --name=lyrics2

