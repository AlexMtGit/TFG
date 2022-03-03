# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

def scale(data):
    index = data.index
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data)
    scaled = scaler.transform(data)
    
    df = pd.DataFrame(scaled, columns = data.columns).set_index(index)
    
    return df
    
def split_train_test(df, train_size_, test_size_, scale=True, verbose=True):
    
    if (scale):
        index = df.index
        
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(df)
        scaled = scaler.transform(df)
        
        df = pd.DataFrame(scaled, columns = df.columns).set_index(index)
    
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Consumo(t)'], axis=1),
                                                        df['Consumo(t)'],
                                                        train_size=train_size_,
                                                        test_size = test_size_,
                                                        random_state = 1, shuffle = False)
    

    if verbose:
        print('Porcentaje datos entrenamiento: ',round(len(X_train)/len(df)*100,2))
        print('Porcentaje datos test: ',round(len(X_test)/len(df)*100,2))
    
        print('\nNúmero de muestras del conjunto de entrenamiento: ',len(X_train))
        print('Ejemplos de Train:')
        print('Variables predictoras:')
        print(X_train) 
        print('Variable objetivo:')
        print(y_train) 
        print('\nNúmero de muestras del conjunto de test: ',len(X_test))
        print('Ejemplos de Test:')
        print('Variables predictoras:')
        print(X_test) 
        print('Variable objetivo:')
        print(y_test) 
    
    
        print('\nNúmero total de muestras',len(X_train)+len(X_test))
        
    return X_train, X_test, y_train, y_test


def regression_results(y_true, y_pred):
    r2=metrics.r2_score(y_true, y_pred)
    print('r2: ', round(r2,2))
    
    
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    print('MAE: ', round(mean_absolute_error,2))
    
    mse=metrics.mean_squared_error(y_true, y_pred)
    print('MSE: ', round(mse,2))
    
    print('RMSE: ', round(np.sqrt(mse),2))
    
    Valor_medio_y_true=np.average(y_true)
    mape=100*mean_absolute_error/Valor_medio_y_true
    print('MAPE (%): ', round(mape,2))
    
    cc = np.corrcoef(y_true, y_pred)[0][1]
    if not np.isnan(cc):
        print('CC: ', round(cc,3))
    
def desnormaliza_salida(df, train_size_, test_size_,y_scaled):
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Consumo(t)'], axis=1),
                                                        df['Consumo(t)'],
                                                        train_size=train_size_,
                                                        test_size = test_size_,
                                                        random_state = 1, shuffle = False)
    
    
    if (len(y_scaled)==len(y_train)):
        index = y_train.index
    
        scaler_output = MinMaxScaler(feature_range=(0,1))
        scaler_output.fit(y_train.values.reshape(-1, 1))
        #no_scaled =scaler_output.inverse_transform(y_scaled.values.reshape(-1, 1))
        no_scaled =scaler_output.inverse_transform(y_scaled.reshape(-1, 1))
        
    elif (len(y_scaled)==len(y_test)):
        index = y_test.index
    
        scaler_output = MinMaxScaler(feature_range=(0,1))
        scaler_output.fit(y_test.values.reshape(-1, 1))
        #no_scaled =scaler_output.inverse_transform(y_scaled.values.reshape(-1, 1))
        no_scaled =scaler_output.inverse_transform(y_scaled.reshape(-1, 1))    
    
    y_no_scaled = pd.DataFrame(no_scaled).set_index(index)
    
    return y_no_scaled
    
def grafica_resultado(model,title,df,train_size_, test_size_,show_scaled=True,show_train=True,show_test=True):
    X_train, X_test, y_train, y_test = split_train_test(df, train_size_, test_size_, scale=False, verbose=False)
    X_train_prep, X_test_prep, y_train_prep, y_test_prep = split_train_test(df, train_size_, test_size_, scale=True, verbose=False)
    
    if show_train:
        x_index = X_train.index
        y_train_pred=model.predict(X_train_prep)
        
        if show_scaled:
            plt.rcParams["figure.figsize"] = (15,4)
            plt.title(title+' - Resultados para Entrenamiento')
            plt.plot(x_index,y_train_prep, label='REAL')
            plt.plot(x_index,y_train_pred, label='PREDICCIÓN')
            plt.legend()
            plt.show()
        else:
            y_pred_no_scaled=desnormaliza_salida(df,train_size_,test_size_,y_train_pred)
            plt.rcParams["figure.figsize"] = (15,4)
            plt.title(title+' - Resultados para Entrenamiento')
            plt.plot(x_index,y_train, label='REAL')
            plt.plot(x_index,y_pred_no_scaled.values, label='PREDICCIÓN')
            plt.legend()
            plt.show()
        
    if show_test:
        x_index = X_test.index
        y_test_pred=model.predict(X_test_prep)
        
        if show_scaled:
            plt.rcParams["figure.figsize"] = (15,4)
            plt.title(title+' - Resultados para Test')
            plt.plot(x_index,y_test_prep, label='REAL')
            plt.plot(x_index,y_test_pred, label='PREDICCIÓN')
            plt.legend()
            plt.show()           
        else:
            y_pred_no_scaled=desnormaliza_salida(df,train_size_,test_size_,y_test_pred)
            plt.rcParams["figure.figsize"] = (15,4)
            plt.title(title+' - Resultados para Test')
            plt.plot(x_index,y_test, label='REAL')
            plt.plot(x_index,y_pred_no_scaled.values, label='PREDICCIÓN')
            plt.legend()
            plt.show()

def crear_tabla_errores_cv_train(df, train_size_, test_size_,modelo,nombre,cv):

    nombre_filas = [('Pliegue ' + str(i)) for i in range(1,11)]

    metricas = ['neg_root_mean_squared_error', 'neg_mean_squared_error', 'neg_median_absolute_error']

    
    X_train, X_test, y_train, y_test = split_train_test(df,train_size_,test_size_,scale=True,verbose=False)

    
    # Obtenemos los ''scores'', donde encontraremos los errores RMSE y MAE
    scores = cross_validate(modelo, X_train, y_train, cv = cv, scoring = metricas)
    
    RMSEs_train = []
    MSEs_train = []
    MAEs_train = []
    # Los añadimos a la lista
    RMSEs_train.append((-1) * scores['test_neg_root_mean_squared_error'])
    MSEs_train.append((-1) * scores['test_neg_mean_squared_error'])
    MAEs_train.append((-1) * scores['test_neg_median_absolute_error'])

    # Creamos un dataframe vacío con columnas los modelos y diez filas, correspondientes a los pliegues
    dfRMSE = pd.DataFrame(0, columns = [nombre], index = nombre_filas)
    # Le añadimos nombre al índice
    dfRMSE.index.name = 'Pliegues'
    # Colocamos los errores correspondientes a cada modelo en su columna
    for i in range(0, 1):
        dfRMSE.iloc[:,i] = RMSEs_train[i]
        
    # Creamos un dataframe vacío con columnas los modelos y diez filas, correspondientes a los pliegues
    dfMSE = pd.DataFrame(0, columns = [nombre], index = nombre_filas)
    # Le añadimos nombre al índice
    dfMSE.index.name = 'Pliegues'
    # Colocamos los errores correspondientes a cada modelo en su columna
    for i in range(0, 1):
        dfMSE.iloc[:,i] = MSEs_train[i]


    # Creamos un dataframe vacío con columnas los modelos y diez filas, correspondientes a los pliegues
    dfMAE = pd.DataFrame(0, columns = [nombre], index = nombre_filas)
    # Le añadimos nombre al índice
    dfMAE.index.name = 'Pliegues'
    # Colocamos los errores correspondientes a cada modelo en su columna
    for i in range(0, 1):
        dfMAE.iloc[:,i] = MAEs_train[i]

    return dfRMSE,dfMSE,dfMAE
