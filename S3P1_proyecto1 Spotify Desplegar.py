#!/usr/bin/env python
# coding: utf-8

# ![image info](https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/images/banner_1.png)

# # Proyecto 1 - Predicción de popularidad en canción
# 
# En este proyecto podrán poner en práctica sus conocimientos sobre modelos predictivos basados en árboles y ensambles, y sobre la disponibilización de modelos. Para su desarrollo tengan en cuenta las instrucciones dadas en la "Guía del proyecto 1: Predicción de popularidad en canción".
# 
# **Entrega**: La entrega del proyecto deberán realizarla durante la semana 4. Sin embargo, es importante que avancen en la semana 3 en el modelado del problema y en parte del informe, tal y como se les indicó en la guía.
# 
# Para hacer la entrega, deberán adjuntar el informe autocontenido en PDF a la actividad de entrega del proyecto que encontrarán en la semana 4, y subir el archivo de predicciones a la [competencia de Kaggle](https://www.kaggle.com/competitions/miad-2025-12-prediccion-popularidad-en-cancion).

# ## Datos para la predicción de popularidad en cancion
# 
# En este proyecto se usará el conjunto de datos de datos de popularidad en canciones, donde cada observación representa una canción y se tienen variables como: duración de la canción, acusticidad y tempo, entre otras. El objetivo es predecir qué tan popular es la canción. Para más detalles puede visitar el siguiente enlace: [datos](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset).

# ## Ejemplo predicción conjunto de test para envío a Kaggle
# 
# En esta sección encontrarán el formato en el que deben guardar los resultados de la predicción para que puedan subirlos a la competencia en Kaggle.

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# In[8]:


# Importación librerías
import pandas as pd
import numpy as np


# ### Disponibilizar Modelo con Flask

# In[5]:


import pandas as pd
import numpy as np
import joblib
from flask import Flask
from flask_restx import Api, Resource, fields, reqparse # reqparse para argumentos

# Carga Modelo
MODEL_FILENAME = 'Popularidad_spotify_LR.joblib'
try:
    pipeline_model = joblib.load(MODEL_FILENAME)
    print(f"Modelo (pipeline) cargado exitosamente desde '{MODEL_FILENAME}'")
    # Columnas soportadas por nuestro modelo
    expected_columns = ['duration_ms', 'danceability', 'energy', 'loudness',
                        'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo', 'explicit', 'key',
                        'mode', 'time_signature']
    print(f"Columnas esperadas por el modelo: {expected_columns}")

except FileNotFoundError:
    print(f"Error: Archivo del modelo '{MODEL_FILENAME}' no encontrado.")
    print("Asegúrate de que el archivo esté en el mismo directorio o proporciona la ruta correcta.")
    pipeline_model = None # Marcar como no cargado
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    pipeline_model = None

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Spotify Popularity Prediction API',
    description='API para predecir la popularidad de una canción usando un modelo simple.'
)

ns = api.namespace('predict',
    description='Predicción de Popularidad')

# Argumentos de Entrada
parser = reqparse.RequestParser()

parser.add_argument('duration_ms', type=float, required=True, help='Duración en ms', location='args')
parser.add_argument('danceability', type=float, required=True, help='Danceability (0-1)', location='args')
parser.add_argument('energy', type=float, required=True, help='Energy (0-1)', location='args')
parser.add_argument('loudness', type=float, required=True, help='Loudness (dB)', location='args')
parser.add_argument('speechiness', type=float, required=True, help='Speechiness (0-1)', location='args')
parser.add_argument('acousticness', type=float, required=True, help='Acousticness (0-1)', location='args')
parser.add_argument('instrumentalness', type=float, required=True, help='Instrumentalness (0-1)', location='args')
parser.add_argument('liveness', type=float, required=True, help='Liveness (0-1)', location='args')
parser.add_argument('valence', type=float, required=True, help='Valence (0-1)', location='args')
parser.add_argument('tempo', type=float, required=True, help='Tempo (BPM)', location='args')
parser.add_argument('explicit', type=int, required=True, help='Explicit (0 o 1)', location='args') # Tratar booleano como 0/1
parser.add_argument('key', type=int, required=True, help='Key (0-11)', location='args')
parser.add_argument('mode', type=int, required=True, help='Mode (0 o 1)', location='args')
parser.add_argument('time_signature', type=int, required=True, help='Time Signature', location='args')

# Lo que la API devolverá
popularity_resource_fields = api.model('PopularityPrediction', {
    'predicted_popularity': fields.Integer, 
})

def predict_popularity(args):
    """Realiza la predicción usando el pipeline cargado."""
    if pipeline_model is None:
        return -1

    try:
        input_data = pd.DataFrame([args], columns=expected_columns)

    except Exception as e:
        print(f"Error al crear DataFrame de entrada: {e}")
        return -2

    try:
        prediction_raw = pipeline_model.predict(input_data)
    except Exception as e:
        print(f"Error durante la predicción del modelo: {e}")
        return -3

    prediction_value = prediction_raw[0]
    prediction_clipped = np.clip(prediction_value, 0, 100) 
    prediction_final = int(prediction_clipped.round()) 

    return prediction_final

@ns.route('/')
class PopularityApi(Resource):

    @ns.expect(parser) 
    @ns.marshal_with(popularity_resource_fields) 
    def get(self):
        """Obtiene los parámetros de la canción y devuelve la predicción de popularidad."""
        args = parser.parse_args()

        predicted_value = predict_popularity(args)

        if predicted_value < 0:
             api.abort(500, f"Error en el servidor durante la predicción (código: {predicted_value})") 

        return {
            "predicted_popularity": predicted_value
        }, 200 

# --- Ejecución de la Aplicación ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


# In[ ]:





# In[ ]:




