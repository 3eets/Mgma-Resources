# Objetivo: Predecir quien de los actuales jugadores en el circuito ATP que están compitiendo a nivel Challenger estarán dentro
#           de los mejores 60 del mundo. 
#
#1) Iniciar las librerias en nuestro notebook e importar
#2) Determinar el enfoque de nuestra muestra de datos al definir la categoria Challenger
#3) Cargar el archivo ChallengerZone al ambiente de trabajo
#4) Preparación y limpieza de datos
#5) Análisis Descriptivo
#6) Dividir el conjunto de datos en conjunto de training y de test
#7) Construir y ajustar el modelo en base al training dataset 
#8) Predecimos que jugadores formaran del top 60 utilizando el modelo con nuevos datos (test dataset)
#9) Validar el comportamiento del modelo
# Iterar los pasos 5, 6, 7, 8, 9 con los siguientes modelos de clasificación:
#       - Regresion de logistica
#       - Random Forest
#       - LightGBM (Gradient Boosting)
#       - Autoencoder 
#       - Semisupervised (Autoencoder + LightGBM)
#10) Redefinir el modelo para conseguir mejores predicciones
#
#
#
# 
# PASO 1) 
# Dentro de databricks podemos instalar librerias de python que funcionan especificamente para nuestro cuaderno
# La ejecucuión tiene que ser en su propia celda

%run Users/andrade.dr@gmail.com/InstallDependencies

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
color= sns.color_palette()
import matplotlib as mpl
%matplotlib inline

from sklearn import preprocessing as pp
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


#
#PASO 2 
#Determinamos por medio de un muestreo bootstrap cual es el ranking medio de un jugador de Challenger y sacamos un intervalo
#de confianza de 99% para trabajar con los jugadores dentro de este rango.

new_calendonia = np.array([74,76,808,156,218,206,413,695,689,221,711,204,96,458,146,566,161,232,279,212,252,82,320,303,208,245,103,255,243,326,240,420])

# Arriba vemos el ranking de los jugadores participando del Challenger New Calendonia 2020 de donde podemos
# sacar 5k muestras (muestreo bootstrap) de esta muestra inicial para aproximarnos a la media de la población real.

np.random.seed(233)

# parametros de bootstrap
n = len(new_calendonia)
reps = 5000

# bootstrap en python es tan simple como ejecutar np.random.choice donde se recogen (reps) muestras con reemplazo de nuestra muestra inicial.

samples = np.random.choice(new_calendonia, (n,reps), replace=True)

# terminamos encontrando la media de cada una de las 5k muestras y encontramos cual es el promedio para sacar nuestro intervalo con un 99%
# de confianza del ranking de un jugador de Challenger (a 3 deviaciones estandar del promedio).

median = np.median(samples, axis=0)
median.sort()
interval = (median.mean() - (median.std()*3), (median.mean()+(median.std()*3)))
interval


#
#PASO 3 cargamos el archivo. 
df = pd.read_csv('/dbfs/FileStore/tables/ChallengerTour.csv', header='infer')

# De acuerdo a nuestro intervalo se ha recogido previamente las estádisticas de los jugadores rankeados entre
# la posición 180 a 300 del mundo según el ranking mundial en Dic. 2019 para procesar sus datos con los datos de los mejores 60 jugadores del mundo.

df.describe().T


# PASO 4 
# Algunas de las variables tiene algunos nombres que vamos renonmbrar para hacer una mejor interpretación.

df = df.rename(columns={'Wins' : 'Ch_Wins', 'Total': 'Ch_Total', 'Finals' : 'Ch_Finals', 'Clay' : 'Ch_Clay', 'Total.Clay' : 'Ch_Total_Clay', 'Hard': 'Ch_Hard', 'Total.Hard' : 'Ch_Total_Hard', 'Grass': 'Ch_Grass', 'Total.Grass': 'Ch_Total_Grass'})

# Además agregamos una variable Label que será nuestra variable objetivo.
df['Label'] = [1 if i <= 60 else 0 for i in df.Ranking]

# PASO 5 Analisis descriptivo. Nuestra dataframe tiene 181 observaciones. Es fácil darnos cuenta que no existen NAs o NULLS.
#
# Revisemos la correlación de las variables con nuestra variable Label () para hacer buena costumbre

plt.figure(figsize = (15,14))
sns.heatmap(df.corr(), linewidths = 0.2, vmax = 1, square = True, linecolor = 'white', annot = True)
plt.show()
plt.gcf().clear()
plt.clear


#Podemos visualizar la distribución de la variable objetivo
df.Label = df['Label'].astype('category')
display(plt.pie(df.Label.value_counts(), labels = df.Label.cat.categories))


# Aparte de los 60 jugadores en el top 60 vemos a los 121 jugadores que representan el rango Challenger Zone entre el 180 y 300 de ranking mundial.
print(df.Label.value_counts())


# También visualizamos la distribución de nuestras variables como parte del análisis descriptivo (en muchos casos para tratar con extremos).

fig = plt.figure(figsize=(15,15)) 
plt.title("Distribución de las variables de predicción (Frecuencia relativa)")
for i in range(1,21):
  ax1 = fig.add_subplot(5,4,i)
  sns.kdeplot(df.iloc[:,i+1], ax=ax1, shade=True)
  if i == 21: 
    display(sns.kdeplot(df.iloc[:,i+1], ax=ax1, shade=True))

# La variable Ch_Total_Grass tiene la correlación más baja con nuestra variable objetivo
plt.gcf().clear()
ceros = df.Ch_Total_Grass[df['Ch_Total_Grass'] == 0].count()
ceros


# Como primer paso vamos a dejar a un lado la variable Ch_Total_Grass por ser una variable con poca correlación y con muchos ceros (63% de nuestra muestra) y para trabajar
# sin extremos (outliers) pasaremos a recortar el 1% de las observaciones de las variables Year.Total, Year.Wins, Ch_Total, Ch_Wins 
# ya que tienen rangos significantes y una distribución asimétrica a la derecha.

q = df['Year.Total'].quantile(0.99)
df_ex = df[df['Year.Total'] < q]

q = df_ex['Year.Wins'].quantile(0.99)
df_ex = df_ex[df_ex['Year.Wins'] < q]

q = df_ex['Ch_Total'].quantile(0.99)
df_ex = df_ex[df_ex['Ch_Total'] < q]

q = df_ex['Ch_Wins'].quantile(0.99)
df_challenger = df_ex[df_ex['Ch_Wins'] < q]

df_challenger = df_challenger.drop('Ch_Total_Grass', axis = 1)

# Ahora vemos que de 181 observaciones hemos pasado a 172 observaciones. 
df_challenger.info()

#
# Para poder ejecutar los modelos con resultados que podamos interpretar aun mas vamos a resumir las variables que equivalen al rendimiento en Challengers
# de todos los jugadores en una variable ChForm para aprovechar la segmentación que nos presta el modelo KNN de nuestras observaciones. 
# Esto con la intención de sacar mejores predicciones y reducir el error.

dataX = df_challenger.drop(['Label','Ranking','Natl.'], axis = 1)
dataY = df_challenger.Label
 
challenger_variables = ['Year' in i for i in dataX.columns]
columns = dataX.columns[challenger_variables]
dataChForm = dataX.drop(columns, axis =1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy = True)
dataX_scaled = scaler.fit_transform(dataChForm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(dataX_scaled, dataY)
proba = knn.predict_proba(dataX_scaled)

proba = pd.DataFrame(proba)
proba.index = dataX.index
dataX['ChForm'] = proba[1]

#
# PASO 6 separamos nuestra dataframe en training y un test set. Por ser una base de datos tan pequeña vamos a dejar al menos 30 observaciones en nuestro test set
# que equivale a un split aproximado de 80% - 20%
#

variablesXescalar = dataX.columns
sX = StandardScaler(copy = True, with_mean= True, with_std=True)
dataX.loc[:,variablesXescalar] = sX.fit_transform(dataX[variablesXescalar])

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = .2, random_state = 223, stratify = dataY)

# Utilizamos random_state para poder reproducir los datos. Además nos aseguramos al usar "stratify" de mantener la proporcion de la variable Label 
# de nuestra dataframe original (la misma proporcion en nuestro test y training set entre jugadores dentro del top 60 y los de Challenger). 


# El set de validación con el cual se hace cross validation nos muestra como esta rindiendo nuestro modelo y se esta entrenando por lo que vamos a dividir nuestro training set
# en más secciones y crear 3 sets de cross validación con la función k-fold. El concepto detras de este método es el mismo por lo cual separamos nuestro dataframe original
# entre training y test set en un principio.

k_fold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 223)

#
# REGRESION LOGISTICA
#

# Primero analizaremos la multicolinearidad entre las variables de predicción y nuestra variable objetivo. No hay regla exacta en las variables que se pueden escoger
# por eso nos quedaremos con las variables que tengan un VIF menor a 15.

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(dataX.values, i) for i in range(dataX.shape[1])]
vif['Variables'] = dataX.columns
vif

# PASO 7
#Definimos los parametros

penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 223
solver = 'liblinear'

logReg = LogisticRegression(penalty = penalty, C=C, class_weight = class_weight, random_state = random_state, solver = solver)
# Entrenamos el modelo

entrenamientoResultados = []
cvResultados = []
prediccionesSetsKFolds = pd.DataFrame(data = [], index = y_train.index, columns = [0,1])
prediccionesSetX_test = pd.DataFrame(data =[], index = y_test.index, columns = [0,1])

model = logReg

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):
  X_train_fold, X_cv_fold = X_train.iloc[train_index,[2,7,10,17,18,19]], X_train.iloc[cv_index, [2,7,10,17,18,19]]
  y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

  model.fit(X_train_fold, y_train_fold)
  loglossTraining = log_loss(y_train_fold, model.predict_proba(X_train_fold)[:,1])
  entrenamientoResultados.append(loglossTraining)
  
  prediccionesSetsKFolds.loc[X_cv_fold.index, :] = model.predict_proba(X_cv_fold)
  loglossCV = log_loss(y_cv_fold, prediccionesSetsKFolds.loc[X_cv_fold.index, 1])
  cvResultados.append(loglossCV)
  
  
  print('Log Loss Entrenamiento: ', loglossTraining)
  print('Log Loss CV: ', loglossCV)

# PASO 8
prediccionesSetX_testReg = pd.DataFrame(data = model.predict_proba(X_test.iloc[:,[2,7,10,17,18,19]]), index = y_test.index, columns = [0,1])


# PASO 9
loglossLogisticRegression = log_loss(y_train, prediccionesSetsKFolds.loc[:,1])
print('Logistic Regression Log Loss: ', loglossLogisticRegression)


# Precision vs Valor de Prediccion Positivo - Curva
preds = pd.concat([y_train, prediccionesSetsKFolds.loc[:,1]], axis = 1)
preds.columns = ['trueLabel', 'prediccion']
prediccionesSetKFoldsRegresion = preds.copy()

precision, vpp, bordes = precision_recall_curve(preds['trueLabel'],preds['prediccion'])

average_precision = average_precision_score(preds['trueLabel'],preds['prediccion'])


plt.step(vpp, precision, color = 'k', alpha = 0.7, where = 'post')
plt.fill_between(vpp, precision, step = 'post', alpha = 0.3, color = 'k')

plt.xlabel('Valor Predictivo Positivo')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))


#El area de nuestra curva ROC nos puede mostrar la sensibilidad de acuerdo al rendimiento del modelo vs la proporcionFalsaPositiva (1-Especifidad)

proporcionFalsaPositiva, sensibilidad, bordes = roc_curve(preds['trueLabel'], preds['prediccion'])
areaCurvaROC = auc(proporcionFalsaPositiva, sensibilidad)

plt.figure()
plt.plot(proporcionFalsaPositiva, sensibilidad, color = 'r', lw=2, label = 'Curva ROC')
plt.plot([0,1], [0,1], color = 'k', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Proporcion Falsa Positiva (1 - Especifidad)')
plt.ylabel('Sensibilidad')
plt.title('ROC:     Area de la Curva ROC = {0:0.2f}'.format(areaCurvaROC))
plt.legend(loc = 'lower right')
plt.show()

#Para poder determinar el rendimiento de nuestro modelo se necesitan aplicar otros modelos a nuestra data y comparar los resultados.

# Por ahora no podemos determinar el rendimiento de nuestro modelo sin tener otros modelos con los que podamos comparar
# Continuamos aplicando el modelo Random Forest a nuestro training y test sets.

#RANDOM FOREST
#Parametros

n_estimators = 50
max_features = 'auto'
max_depth = 5
min_samples_split = 5
min_samples_leaf = 3
min_weight_fraction_leaf = 0.0
max_leaf_nodes = None
bootstrap = True
oob_score = False
n_jobs = -1
random_state = 223
class_weight = 'balanced'

RFC = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_leaf_nodes = max_leaf_nodes, bootstrap = bootstrap, oob_score= oob_score, n_jobs = n_jobs, random_state = random_state, class_weight = class_weight)

# Segun nuestra configuracion haremos 50 ejecuciones y sacaremos el promedio de los resultados de todos los arboles. Se consideran para cada arbol la raiz cuadrada de las 19 variables de prediccion que es 'auto' 4 variables por arbol. En max_depth None permite al arbol crecer sin limites y separandose lo mas posible. Por ultimo, random_state es igual a 223 para poder reproducir los mismos resultados ademas de indicar que nuestra muestra no esta balanceada y que se necesita dar mas peso a las observaciones con un Label de 1 (class_weight = balanced).
#Utilizaremos k-folds igual (3 divisiones del training set) para trabajar con mas sets de prueba y cross validar el rendimiento.

entrenamientoResultados = []
cvResultados = []
prediccionesSetsKFolds = pd.DataFrame(data = [], index = y_train.index, columns = [0,1])

model = RFC

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):
  X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:]
  y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
  
  model.fit(X_train_fold, y_train_fold)
  loglossTraining = log_loss(y_train_fold, model.predict_proba(X_train_fold)[:,1])
  entrenamientoResultados.append(loglossTraining)
  
  prediccionesSetsKFolds.loc[X_cv_fold.index,:] = model.predict_proba(X_cv_fold)
  loglossCV = log_loss(y_cv_fold, prediccionesSetsKFolds.loc[X_cv_fold.index,1])
  cvResultados.append(loglossCV)
  
  
  
  print('Log Loss Entrenamiento : ', loglossTraining)
  print('Log Loss CV: ', loglossCV)

prediccionesSetX_testRF = pd.DataFrame(data = model.predict_proba(X_test), index = y_test.index, columns = [0,1])
loglossRandomForest = log_loss(y_train, prediccionesSetsKFolds.loc[:,1])
print('Log Loss Random Forest: ', loglossRandomForest)

# Podemos observar que ya nuestra funcion de costo nos muestra un valor menor por mas de la mitad que nuestro modelo de Regresion Logistica.
# Como hemos hecho para el modelo anterior buscaremos visualizar el promedio de precision y la curva ROC

# Precision vs Valor de Prediccion Positivo - Curva
preds = pd.concat([y_train, prediccionesSetsKFolds.loc[:,1]], axis = 1)
preds.columns = ['trueLabel', 'prediccion']
prediccionesSetKFoldsRF = preds.copy()

precision, vpp, bordes = precision_recall_curve(preds['trueLabel'],preds['prediccion'])

average_precision = average_precision_score(preds['trueLabel'],preds['prediccion'])


plt.step(vpp, precision, color = 'k', alpha = 0.7, where = 'post')
plt.fill_between(vpp, precision, step = 'post', alpha = 0.3, color = 'k')

plt.xlabel('Valor Predictivo Positivo')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))

#El area de nuestra curva ROC nos puede mostrar la sensibilidad de acuerdo al rendimiento del modelo vs la proporcionFalsaPositiva (1-Especifidad)

proporcionFalsaPositiva, sensibilidad, bordes = roc_curve(preds['trueLabel'], preds['prediccion'])
areaCurvaROC = auc(proporcionFalsaPositiva, sensibilidad)

plt.figure()
plt.plot(proporcionFalsaPositiva, sensibilidad, color = 'r', lw=2, label = 'Curva ROC')
plt.plot([0,1], [0,1], color = 'k', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Proporcion Falsa Positiva (1 - Especifidad)')
plt.ylabel('Sensibilidad')
plt.title('ROC:     Area de la Curva ROC = {0:0.2f}'.format(areaCurvaROC))
plt.legend(loc = 'lower right')
plt.show()

#Los resultados son notables y vale aplicar un modelo de supervacion mas para encontrar como se comparan


#LightGBM

#Parametros
params_lightGB = {'task': 'train', 'application':'binary', 'num_class':1, 'boosting':'gbdt', 'objective':'binary', 'metric':'binary_logloss', 'metric_freq':50, 'is_training_metric':False, 'max_depth':4, 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'bagging_seed': 223, 'verbose': -1, 'num_threads': 16}

entrenamientoResultados = []
cvResultados = []
prediccionesSetsKFolds = pd.DataFrame(data = [], index = y_train.index, columns = ['prediccion'])
y_train = y_train.astype('int')

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):
  X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:]
  y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
  
  lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
  lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference = lgb_train)
  gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=2000, valid_sets = lgb_eval, early_stopping_rounds = 200)
  
  loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold, num_iteration=gbm.best_iteration))
  entrenamientoResultados.append(loglossTraining)
  
  prediccionesSetsKFolds.loc[X_cv_fold.index, 'prediccion'] = gbm.predict(X_cv_fold, num_iteration = gbm.best_iteration)
  loglossCV = log_loss(y_cv_fold, prediccionesSetsKFolds.loc[X_cv_fold.index, 'prediccion'])
  cvResultados.append(loglossCV)
  
 
  
  print('Log Loss Entrenamiento: ', loglossTraining)
  print('Log Loss CV: ', loglossCV)

lgb_test = lgb.Dataset(X_test, y_test, reference = lgb_train) 
prediccionesSetX_testLightGB = pd.DataFrame(data = gbm.predict(X_test, num_iteration = gbm.best_iteration), index = y_test.index, columns = ['prediccion'])
loglossLightGBM = log_loss(y_train, prediccionesSetsKFolds.loc[:,'prediccion'])
print('LightGBM Log Loss: ', loglossLightGBM)


# Podemos observar que ya nuestra funcion de costo nos muestra un valor menor por mas de la mitad que nuestro modelo de Regresion Logistica.
# Como hemos hecho para el modelo anterior buscaremos visualizar el promedio de precision y la curva ROC

# Precision vs Valor de Prediccion Positivo - Curva
preds = pd.concat([y_train, prediccionesSetsKFolds.loc[:,'prediccion']], axis = 1)
preds.columns = ['trueLabel', 'prediccion']
prediccionesSetKFoldsLightGB = preds.copy()

precision, vpp, bordes = precision_recall_curve(preds['trueLabel'],preds['prediccion'])

average_precision = average_precision_score(preds['trueLabel'],preds['prediccion'])


plt.step(vpp, precision, color = 'k', alpha = 0.7, where = 'post')
plt.fill_between(vpp, precision, step = 'post', alpha = 0.3, color = 'k')

plt.xlabel('Valor Predictivo Positivo')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))

#El area de nuestra curva ROC nos puede mostrar la sensibilidad de acuerdo al rendimiento del modelo vs la proporcionFalsaPositiva (1-Especifidad)

proporcionFalsaPositiva, sensibilidad, bordes = roc_curve(preds['trueLabel'], preds['prediccion'])
areaCurvaROC = auc(proporcionFalsaPositiva, sensibilidad)

plt.figure()
plt.plot(proporcionFalsaPositiva, sensibilidad, color = 'r', lw=2, label = 'Curva ROC')
plt.plot([0,1], [0,1], color = 'k', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Proporcion Falsa Positiva (1 - Especifidad)')
plt.ylabel('Sensibilidad')
plt.title('ROC:     Area de la Curva ROC = {0:0.2f}'.format(areaCurvaROC))
plt.legend(loc = 'lower right')
plt.show()

# Continuamos con nuestro unico modelo unsupervisado

#AUTOENCODER

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from keras.layers import BatchNormalization, Input, Lambda
from keras import regularizers
from keras.losses import mse, binary_crossentropy

def resultadosAnomalos(DForiginal, DFreducido):
  error = np.sum((np.array(DForiginal) - np.array(DFreducido))**2, axis = 1)
  error = pd.Series(data = error, index = DForiginal.index)
  error = (error-np.min(error))/(np.max(error) - np.min(error))
  return error


test_resultados = []
for i in range(0,5):
  model = Sequential()

  model.add(Dense(units = 25, activation = 'linear', activity_regularizer = regularizers.l1(10e-5), input_dim=20, name = 'hidden_layer'))
  model.add(Dropout(0.02))
  model.add(Dense(units = 20, activation = 'linear'))
  
  model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
  
  history = model.fit(x = X_train, y = X_train, epochs = 5, batch_size = 32, shuffle = True, validation_split = .2, verbose = 1)
  predicciones = model.predict(X_test, verbose = 1)
  resultadosAnom = resultadosAnomalos(X_test, predicciones)
  preds = pd.concat([y_test, resultadosAnom], axis =1)
  preds.columns = ['LabelOriginal', 'ResultadosAnomalos']
  precision, vpp, bordes = precision_recall_curve(preds['LabelOriginal'], preds['ResultadosAnomalos'])
  precisionPromd = average_precision_score(preds['LabelOriginal'], preds['ResultadosAnomalos'])
  PrediccionesAutoEncoder = preds.copy()
  
  test_resultados.append(precisionPromd)
  plt.step(vpp, precision, color = 'k', alpha= 0.7, where = 'post')
  plt.fill_between(vpp, precision, step = 'post', alpha = 0.3, color = 'k')
  
  plt.xlabel('ValorProporcionPositiva')
  plt.ylabel('Precision')
  plt.ylim([0.0,1.05])
  plt.xlim([0.0,1.0])
  
  plt.title('Curva Precision-VPP:    Precision Promedio = {0:0.2f}'.format(precisionPromd))
  
  proporcionFalsaPositiva, sensibilidad, bordes = roc_curve(preds['LabelOriginal'], preds['ResultadosAnomalos'])
  areaCurvaROC = auc(proporcionFalsaPositiva, sensibilidad)
  
  plt.figure()
  plt.plot(proporcionFalsaPositiva, sensibilidad, color = 'r', lw=2, label = 'Curva ROC')
  plt.plot([0,1], [0,1], color = 'k', lw = 2, linestyle = '--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Proporcion Falsa Positiva (1 - Especifidad)')
  plt.ylabel('Sensibilidad')
  plt.title('ROC:     Area de la Curva ROC = {0:0.2f}'.format(areaCurvaROC))
  plt.legend(loc = 'lower right')
  plt.show()



print("Promedio de la Precision promedio de 5 ejecuciones: ", np.mean(test_resultados))
test_resultados


#
# Para hacer un modelo semisupervisado vamos a utilizar la capa del autoencoder y pasar a nuestras variables
# con las cuales aplicamos LigthGB que fue el mejor modelo supervisado que hemos revisado.

X_train = X_train.copy()

nombre_capa = 'hidden_layer'
capa_intermedia_modelo = Model(inputs=model.input, outputs = model.get_layer(nombre_capa).output)
output_intermedio_train = capa_intermedia_modelo.predict(X_train)
output_intermedio_test = capa_intermedia_modelo.predict(X_test)

output_intermedio_trainDF = pd.DataFrame(data= output_intermedio_train, index = X_train.index)
output_intermedio_testDF = pd.DataFrame(data= output_intermedio_test, index = X_test.index)

X_train = X_train.merge(output_intermedio_trainDF, left_index = True, right_index = True)
X_test = X_test.merge(output_intermedio_testDF, left_index = True, right_index = True)

# Hay 45 variables

entrenamientoResultados = []
cvResultados = []
prediccionesSetsKFolds = pd.DataFrame(data = [], index = y_train.index, columns = ['prediccion'])
y_train = y_train.astype('int')

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):
  X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:]
  y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
  
  lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
  lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference = lgb_train)
  gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=5000, valid_sets = lgb_eval, early_stopping_rounds = 200)
  
  loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold, num_iteration=gbm.best_iteration))
  entrenamientoResultados.append(loglossTraining)
  
  prediccionesSetsKFolds.loc[X_cv_fold.index, 'prediccion'] = gbm.predict(X_cv_fold, num_iteration = gbm.best_iteration)
  loglossCV = log_loss(y_cv_fold, prediccionesSetsKFolds.loc[X_cv_fold.index, 'prediccion'])
  cvResultados.append(loglossCV)
  
  print('Log Loss Entrenamiento: ', loglossTraining)
  print('Log Loss CV: ', loglossCV)

lgb_test = lgb.Dataset(X_test, y_test, reference = lgb_train) 
prediccionesSetX_testSemi = pd.DataFrame(data = gbm.predict(X_test, num_iteration = gbm.best_iteration), index = y_test.index, columns = ['preddicion'])
loglossLightGBM = log_loss(y_train, prediccionesSetsKFolds.loc[:,'prediccion'])
print('LightGBM Log Loss: ', loglossLightGBM)


preds = pd.concat([y_train, prediccionesSetsKFolds.loc[:,'prediccion']], axis = 1)
preds.columns = ['trueLabel', 'prediccion']
prediccionesSetKFoldsSemi = preds.copy()

precision, vpp, bordes = precision_recall_curve(preds['trueLabel'],preds['prediccion'])

average_precision = average_precision_score(preds['trueLabel'],preds['prediccion'])


plt.step(vpp, precision, color = 'k', alpha = 0.7, where = 'post')
plt.fill_between(vpp, precision, step = 'post', alpha = 0.3, color = 'k')

plt.xlabel('Valor Predictivo Positivo')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))

#El area de nuestra curva ROC nos puede mostrar la sensibilidad de acuerdo al rendimiento del modelo vs la proporcionFalsaPositiva (1-Especifidad)

proporcionFalsaPositiva, sensibilidad, bordes = roc_curve(preds['trueLabel'], preds['prediccion'])
areaCurvaROC = auc(proporcionFalsaPositiva, sensibilidad)

plt.figure()
plt.plot(proporcionFalsaPositiva, sensibilidad, color = 'r', lw=2, label = 'Curva ROC')
plt.plot([0,1], [0,1], color = 'k', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Proporcion Falsa Positiva (1 - Especifidad)')
plt.ylabel('Sensibilidad')
plt.title('ROC:     Area de la Curva ROC = {0:0.2f}'.format(areaCurvaROC))
plt.legend(loc = 'lower right')
plt.show()

variablesPrincipales = pd.DataFrame(data = list(gbm.feature_importance()), index = X_train.columns, columns = ['Importancia'])
variablesPrincipales = variablesPrincipales / variablesPrincipales.sum()
variablesPrincipales.sort_values(by= 'Importancia', ascending = False, inplace = True)
variablesPrincipales


# Vamos a reestablecer nuestra dataframe primero juntando las predicciones de los modelos con el training set (estos son solo los resultados que hemos obtenido 
# en cada modelo que hemos aplicado).
#
predicciones = pd.concat([y_train, prediccionesSetKFoldsRegresion.loc[:,'prediccion'], prediccionesSetKFoldsRF.loc[:,'prediccion'], prediccionesSetKFoldsLightGB.loc[:,'prediccion'], prediccionesSetKFoldsSemi.loc[:,'prediccion']], axis = 1)
predicciones.columns = ['TrueLabel', 'prediccionLogReg', 'prediccionRF', 'prediccionLightGB', 'prediccionSemi']
datasetoutput = df_challenger.merge(predicciones, how = 'left', left_index = True, right_index = True)

# Por último juntamos las predicciones del test set para tener la tabla completa.

prediccionesX_test = pd.concat([y_test, prediccionesSetX_testReg.iloc[:,1], prediccionesSetX_testRF.iloc[:,1], prediccionesSetX_testLightGB, prediccionesSetX_testSemi], axis = 1)
prediccionesX_test.columns = ['TrueLabel', 'prediccionLogReg', 'prediccionRF', 'prediccionLightGB', 'prediccionSemi']
datasetoutput.loc[prediccionesX_test.index,['TrueLabel', 'prediccionLogReg', 'prediccionRF', 'prediccionLightGB', 'prediccionSemi']] = prediccionesX_test.values
potential = datasetoutput[(datasetoutput.Ranking > 180) & (datasetoutput.prediccionLightGB > .4)]
potential.head()

#
# PASO 10
# Hay un factor que nos puede ayudar a sacar mejores resultados. De todos el más alcanzable esta en utilizar los datos actuales.
# Como vemos al ejecutar el codigo en esta linea hay tan solo 43 paises que se pueden categorizar por continente 
#
countries = df_challenger['Natl.'].unique()
len(countries)
# 
# En este caso agregamos la variable continente que podemos usar para filtrar nuestras predicciones.

Europe = ['ESP','SRB','SUI','AUT','GRE','RUS','GER','ITA','FRA','BEL','BUL','GEO','CRO','GBR','MDA','NOR','KAZ','SLO','POR','CZE','SVK','NED','LAT','UKR','SWE','BIH','POL']
Asia = ['JPN','CHN','TPE','KOR','IND','TUR']
Africa = ['TUN']
Oceania = ['AUS']
America = ['USA','URU','ARG','CAN','CHI','BRA','COL','DOM','ECU']

# Agregamos la variable Continent a nuestra base de datos
datasetoutput['Continent'] = ["Europe" if str(i) in Europe else "Asia" if str(i) in Asia else "Africa" if str(i) in Africa else "Oceania" if str(i) in Oceania else "America" for i in datasetoutput['Natl.']]
datasetoutput.Continent = datasetoutput['Continent'].astype('category')
datasetoutput.Continent.dtype
