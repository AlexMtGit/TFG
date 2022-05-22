# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

import scipy.stats as stats;
import scikit_posthocs as scp
import statsmodels as st
import operator

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

def split_train_test_deep(df, train_size_, test_size_, target="Consumo(t)" ,_scale=True, verbose=True):
    if _scale:
        df = scale(df)
        
    n=len(df)
    train_df = df[0:int(n*train_size_)]
    test_df = df[int(n*train_size_):]
    
    features = df.columns.tolist()
    features.remove(target)
    
    X_train, y_train = list(), list()
    for index, row in train_df.iterrows():
        X_train.append([item for item in row[features].tolist()])
        y_train.append(row[target])
        
    X_test, y_test = list(), list()
    for index, row in test_df.iterrows():
        X_test.append([item for item in row[features].tolist()])
        y_test.append(row[target])
        
    if verbose:
        print('Porcentaje datos entrenamiento: ',round(len(X_train)/len(df)*100,2))
        print('Porcentaje datos test: ',round(len(X_test)/len(df)*100,2))
    
        print('\nNúmero de muestras del conjunto de entrenamiento: ',len(X_train))
        print('Ejemplos de Train:')
        print('Variables predictoras:' + str(features))
        print(X_train[0]) 
        print('Variable objetivo:' + target)
        print(y_train[0]) 
        print('\nNúmero de muestras del conjunto de test: ',len(X_test))
        print('Ejemplos de Test:')
        print('Variables predictoras:' + str(features))
        print(X_test[0]) 
        print('Variable objetivo:' + target)
        print(y_test[0]) 
    
    
        print('\nNúmero total de muestras',len(X_train)+len(X_test))
        
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


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
    
    try:
        cc = np.corrcoef(y_true, y_pred)[0][1]
    except:
        cc = np.nan
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
            
def get_MAPE(model, X_val, y_true, epsilon = 0.00000001):
    y_pred = model.predict(X_val)
    ii = 0
    for i in y_true:
        if (i < epsilon) & (i > -epsilon):
             y_true[ii] = epsilon
        else:
             y_true[ii] = y_true[ii]
        ii = ii+1
        
    MAPE = (100/len(y_true))*np.sum(np.abs((y_true - y_pred)/y_true))
    return MAPE      
        

def crear_tabla_errores_cv_train(best_estimator, X_train, y_train, cv, nombre):

    nombre_filas = [('Pliegue ' + str(i)) for i in range(1,11)]

    metricas = {'nrmse' : 'neg_root_mean_squared_error', 'negmse' : 'neg_mean_squared_error',
                'nmae' : 'neg_median_absolute_error', 'MAPE' : get_MAPE }

    
    # Obtenemos los ''scores'', donde encontraremos los errores RMSE y MAE
    scores = cross_validate(best_estimator, X_train, y_train, cv = cv, scoring = metricas)
    
    RMSEs_train = []
    MSEs_train = []
    MAEs_train = []
    MAPEs_train = []
    # Los añadimos a la lista
    RMSEs_train.append((-1) * scores['test_nrmse'])
    MSEs_train.append((-1) * scores['test_negmse'])
    MAEs_train.append((-1) * scores['test_nmae'])
    MAPEs_train.append(scores['test_MAPE'])

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
        
    # Creamos un dataframe vacío con columnas los modelos y diez filas, correspondientes a los pliegues
    dfMAPE = pd.DataFrame(0, columns = [nombre], index = nombre_filas)
    # Le añadimos nombre al índice
    dfMAPE.index.name = 'Pliegues'
    # Colocamos los errores correspondientes a cada modelo en su columna
    for i in range(0, 1):
        dfMAPE.iloc[:,i] = MAPEs_train[i]

    return dfRMSE,dfMAE,dfMAPE

def pairwise_test(df, parametric = True, method = 'bonferroni', decreasing = True, ties=True):
    # Función que devuelve la matriz de rankings.
    # El dataframe de entrada df debe tener los modelos en las columnas y 
    # las medidas en las filas. 
    # parametric = indica si se aplica un test paramétrico (ttest_rel) o no paramétrico (wilcoxon)
    # decreasing = indica el sentido en el que es mejor la medida utilizada. 
    # ties = indca si se incluyen los empates en el resultado final 
    
    if decreasing:
        relate = operator.lt
    else:
        relate = operator.gt
    
    if parametric:
        test = stats.ttest_rel
    else:
        test = stats.wilcoxon
        
    cols = df.columns
    pvalues = np.array([])
    significant = np.array([])
    ranks = pd.DataFrame({"wins":np.zeros(len(cols)),
                         "ties":np.zeros(len(cols)),
                         "losses":np.zeros(len(cols))})
    ranks.index= cols
    for index1 in range(len(cols)):
        for index2 in range(index1+1,len(cols)):
            T,p = test(df.iloc[:,index1],df.iloc[:,index2])
            #print(p)
            #sig,padjust,a1,a2 = st.stats.multitest.multipletests(p,alpha = 0.05,method = method)
            #print(index1,"--",index2,"-->",padjust)
            pvalues = np.append(pvalues,p)
    significant,padjust,a1,a2 = st.stats.multitest.multipletests(pvalues,alpha = 0.05,method = method)
    # print(significant, padjust)
    pos = 0;
    for index1 in range(len(cols)):
        for index2 in range(index1+1,len(cols)):
            #print(significant[pos],"  ",padjust[pos])
            if not significant[pos]:
             #   print("Tie")
                ranks.iloc[index1,1] = ranks.iloc[index1,1]+1
                ranks.iloc[index2,1] = ranks.iloc[index2,1]+1
            else:  
                if relate(df.iloc[:,index1].mean(),df.iloc[:,index2].mean()):
              #      print(index1," wins")
                    ranks.iloc[index1,0] = ranks.iloc[index1,0]+1
                    ranks.iloc[index2,2] = ranks.iloc[index2,2]+1
                else:
               #     print(index2," loses")
                    ranks.iloc[index2,0] = ranks.iloc[index2,0]+1
                    ranks.iloc[index1,2] = ranks.iloc[index1,2]+1
            pos = pos + 1
                
    ranks["diff"] = ranks.wins-ranks.losses
    #ranks = ranks.sort_values(by="diff",ascending = False)
    if  not ties:
        ranks.drop('ties',axis = 'columns', inplace = True)
    return ranks.astype(int) 