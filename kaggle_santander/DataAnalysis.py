
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[2]:


def missing_values_table(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Valores ausentes', 1 : '% do valor total'})
    return mis_val_table_ren_columns.loc[(mis_val_table_ren_columns!=0).any(axis=1)]


# In[3]:


#load data
df = pd.read_csv("data/train.csv")


# ## Scatter plot geral - validar se os dados se apresentam de forma esparça

# In[4]:


df


# In[5]:


#scatter_matrix(df, figsize=(20, 20), diagonal='kde')


# ## Avaliar quantidade de Missing Values

# In[6]:


miss = missing_values_table(df)


# In[7]:


miss = miss.sort_values(by=["Valores ausentes"], ascending=False)


# In[8]:


miss


# In[9]:


plt.plot(sorted(miss['% do valor total'].values))
plt.grid(True)
plt.ylabel("% Valores ausentes")
plt.show()
miss['Valores ausentes'].hist(bins=15, grid=False)
plt.ylabel("Valores ausentes")
plt.xticks([])
plt.show()


# In[10]:


df[miss.index].dtypes


# De acordo com alguns estudos, estimar os Missing values em um conjunto de dados onde o valor ultrapassa 40%, existe uma grande chance de prejudicar o aprendizado.
# Desse modo, os atributos abaixo cerca de 5, estão a partir de 47% até 99%. Sendo assim podemos partir por dois caminhos. Eliminar do conjunto de dados, ou estudar cada atributo e tentar substituir os valores ausentes por um valor que represente essa ausência.
# 
# Atributo  | Valores ausentes | % do valor total | Descrição
# --- | --- | --- | ---
# PoolQC  |	1453 |	`99.520` | Pool quality
# MiscFeature  |	1406 |	`96.301` | Miscellaneous feature not covered in other categories
# Alley  |	1369 |	`93.767` | Type of alley access
# Fence  |	1179 |	80.753 | Fence quality
# FireplaceQu  |	690 |	47.260 | Fireplace quality
# 
# Sendo assim, vou estudar os 5 atributos com o objetivo de substituir os valores ausentes.

# In[11]:


for c in df[['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']].columns:
    print(c,df[c].unique())


# 3 atributos estão relacionados a qualidade, 1 relacionado ao tipo de acesso e outro serve como 'outros', dessa forma os atributos que são relacionados a qualidade, apesar de categóricos, estão dentro de uma escala de distância. Os demais não possuem essa escala.

# In[12]:


df.PoolQC.fillna('NtE', inplace=True)
df.Fence.fillna('NtE', inplace=True)
df.FireplaceQu.fillna('NtE', inplace=True)
df.MiscFeature.fillna('None', inplace=True)
df.Alley.fillna('None', inplace=True)


# In[75]:


col_n = []
col_c = []
for c in miss.index:
    #print(c,df[c].dtypes,df[c].unique())
    if df[c].dtypes == 'object':
        col_c.append(c)
    else:
        col_n.append(c)


# Para as os demais, existem outras abordagens como substituição por vizinhos mais próximos dentre outras, mas para esse experimento vou utilizar a substituição pelos valores médios ou moda

# In[42]:


from sklearn.impute import SimpleImputer


# In[43]:


imputerN = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputerC = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')


# In[ ]:


imputerN = imputerN.fit(df[col_n])
df[col_n] = imputerN.transform(df[col_n])
imputerC = imputerC.fit(df[col_c])
df[col_c] = imputerC.transform(df[col_c])


# ## tratar os tipos object

# In[67]:


df.get_dtype_counts()


# In[74]:


for c in df.columns:
    if df[c].dtypes == 'object':
        print(c,df[c].unique())


# Para os dados categóricos existem diversas abordagens, a mais comum é a conversão para pseudo-inteiros ou a one-hot.

# In[76]:


for c in df.columns:
    if df[c].dtype.name == "object":
        df[c] = df[c].astype('category')


# In[78]:


#conversao para pseudo atributos inteiros
for c in df.columns:
    if df[c].dtypes.name == 'category':
        df[c] = df[c].cat.codes


# In[503]:


def pre_process(df):
    df.PoolQC.fillna('NtE', inplace=True)
    df.Fence.fillna('NtE', inplace=True)
    df.FireplaceQu.fillna('NtE', inplace=True)
    df.MiscFeature.fillna('None', inplace=True)
    df.Alley.fillna('None', inplace=True)
    
    miss = missing_values_table(df)
    
    col_n = []
    col_c = []
    for c in miss.index:
        #print(c,df[c].dtypes,df[c].unique())
        if df[c].dtypes == 'object':
            col_c.append(c)
        else:
            col_n.append(c)
            
    imputerN = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputerC = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    
    imputerN = imputerN.fit(df[col_n])
    df[col_n] = imputerN.transform(df[col_n])
    imputerC = imputerC.fit(df[col_c])
    df[col_c] = imputerC.transform(df[col_c])
    
    for c in df.columns:
        if df[c].dtype.name == "object":
            df[c] = df[c].astype('category')
            
    #conversao para pseudo atributos inteiros
    for c in df.columns:
        if df[c].dtypes.name == 'category':
            df[c] = df[c].cat.codes
    
    return df


# ## Visualização

# In[85]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import seaborn as sn


# In[108]:


df2 = df[df.columns.difference(['SalePrice','Id'])].copy()


# In[119]:


def pca_view(df2, n_components=4):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(df2)
    X = pca.transform(df2)
    print("Soma dos 3 primeiros componentes:",np.sum(pca.explained_variance_ratio_[0:3]))
    
    sn.barplot(list(range(1,len(pca.components_)+1)), 1*pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), )
    plt.show()
    
    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=18, azim=134)
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Componente Principal 1', fontsize = 15)
    ax.set_ylabel('Componente Principal 2', fontsize = 15)

    ax.scatter(X[:, 0], X[:, 1], s = 50)
    plt.show()
    
    return X


# In[541]:


df2.boxplot(figsize=(20, 20), rot=90)


# #### Em alguns atributos, é possível dectectar a presença de outiliers, nesse caso é necessário um pouco de atenção e verificar a necessidade de normalizar os dados.

# In[114]:


from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, QuantileTransformer, MinMaxScaler


# In[354]:


x_st = StandardScaler().fit_transform(df2)


# In[380]:


print("Normalizados:")
pca_view(x_st, 10) # normalizado
print("Dados originais:")
pca_view(df2, 10) # dados reais


# ### Aparentemente a disposição dos dados está mais para não linear do que uma apresentação linear, conforme visto anteriormente na técnica PCA, vou confirmar isso também utilizando um SVM sem kernel.

# In[455]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import sklearn
import xgboost


# In[223]:


X = df2
y = df['SalePrice'].values


# In[387]:


X_train, X_test, y_train, y_test = train_test_split(df2, y, train_size=0.75, test_size=0.25)


# In[388]:


svm_lin = LinearSVR(max_iter=20000, epsilon=0.1)
svm_lin.fit(X_train, y_train)
print(svm_lin.score(X_test, y_test)) 
print("R2",cross_val_score(svm_lin, X_test, y_test, cv=5, scoring='r2'))


# ## RandomForest

# In[457]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
print("R2",cross_validate(rf, X, y, cv=5, scoring=('r2')))


# ## GradientBoosting

# In[458]:


gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
print(gb.score(X_test, y_test))
print("R2",cross_validate(gb, X, y, cv=5, scoring=('r2')))


# ## XGBoost

# In[459]:


xgb = xgboost.XGBRegressor()
xgb.fit(X_train, y_train)
print(xgb.score(X_test, y_test))
print("R2",cross_validate(xgb, X, y, cv=5, scoring=('r2')))


# ## MLP

# In[460]:


mlp = MLPRegressor(hidden_layer_sizes=(200,), max_iter=1000, alpha=0.5)
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))
print("R2",cross_validate(mlp, X, y, cv=5, scoring=('r2')))


# #### Até agora, as técnicas XGBoost, GB, RF e SVM linear estão apresentando uma melhor performance

# A próxima abordagem é ajustar os hiper parâmetros, geralmente o comum é utilizar um técnica de GridSearch.

# In[419]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[475]:


params = {
        'min_child_weight': [1, 5, 10, 20, 30],
        'gamma': [0.5, 1, 1.5, 2, 5, 10, 20, 30, 40, 50, 90, 100, 150],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10, 15, 50]
        }
param_comb = 5


# In[482]:


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[483]:


random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='r2', n_jobs=4, cv=5, random_state=1001 )


# In[484]:


random_search.fit(X_train, y_train)


# In[485]:


print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (5, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)


# In[486]:


results_hp = cross_validate(random_search.best_estimator_, X, y, cv=5, scoring=('r2'))


# In[487]:


results_nm = cross_validate(xgb, X, y, cv=5, scoring=('r2'))


# In[488]:


print("XGB BH:", np.mean(results_hp['test_score']))
print("XGB NM:", np.mean(results_nm['test_score']))


# In[537]:


fig, ax = plt.subplots(figsize=(20, 20))
xgboost.plot_importance(random_search.best_estimator_, ax=ax)


# In[512]:


test_df = pd.read_csv('data/test.csv')


# In[513]:


test_df = pre_process(test_df)


# In[514]:


test_df2 = test_df[test_df.columns.difference(['Id'])].copy()


# In[515]:


pca_view(test_df2)


# In[539]:


y_test = xgb.predict(test_df2)


# In[540]:


results_df = pd.DataFrame(data={'Id':test_df['Id'], 'SalePrice':y_test})
results_df.to_csv('submission-xgb-01.csv', index=False)

