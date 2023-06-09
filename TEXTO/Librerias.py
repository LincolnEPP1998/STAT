!pip install pysentimiento
!pip install torch==2.0.1
!pip uninstall -y transformers accelerate
!pip install transformers accelerate

!pip install flair
!pip install pandas
!pip install numpy
!pip install re
!pip install nltk
!pip install matplotlib
!pip install wordcloud
!pip install PIL
!pip install flair
!pip install spacy
!pip install collections

import nltk
nltk.download('book')
# librerias necesarias

# librerias para manejo de DataFrame y arrays
import pandas as pd
import numpy as np

# librerias para Procesamiento del Lenguaje Natural
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.book import *

# librerias para visualizaciones
import matplotlib.pyplot as plt
from matplotlib import style
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

# librerias para ner

from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
from spacy import displacy
from collections import Counter


# librerias para Analisis de sentimientos

from pysentimiento import create_analyzer
