#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)


df = pd.read_csv('data.csv', sep=';')
""" Análises Iniciais """
# nomes das colunas
print(df.columns)
# primeiros dados
print(df.head())
# informacões de tipos, quantidade de nulos etc...
print(df.info())
# análises básicas das distribuicão destes dados
print(df.describe())
""" Relacões das correlacões """
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.pairplot(df, hue='survived')
sns.distplot(df['pclass'], label='Social Class Distribution')
sns.countplot(x='survived', data=df, hue='sex')
sns.boxplot(x='pclass', y='age', data=df)
""" 
    1) Aqui tomamos algumas decisões, vamos remover os seguintes campos
     -> Cabine não nos interessa pois existem poucos dados
     -> Boat não nos interessa
     -> Destino removemos de proposito
     -> Número do ticket não nos interessa
    2) Mulheres e criancas tem maior chance de sobrevivencia
    3) A classe economica, sexo, idade tem forte correlacão positiva para sobrevivencia
"""
df = df.drop(['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], axis=1)
""" Tratamentos """
def numeric_age(row):
    if type(row['age']) is str:
        return round(float(row['age'].replace(',', '.')), 0)
    return round(float(row['age']), 0)


def handle_age(row):
    age = row['age']
    sex = row['sex']
    pclass = row['pclass']
    if np.isnan(age):
       mean_age_pclass = df.loc[df['pclass'] == pclass]['age'].mean()
       mean_age_sex = df.loc[df['sex'] == sex]['age'].mean()
       age = mean_age_pclass + mean_age_sex / 2
    return round(age, 0)


def handle_fare(row):
    fare = row['fare']
    if type(fare) is str:
        fare = fare.replace(',', '.')
        return round(float(fare), 2)
    return round(float(fare), 2)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
# transformando a tipagem do sexo de Female Male para 1, 0
df['sex'] = pd.get_dummies(df['sex'], dtype=int)
# transformando par 0, 1
df['embarked'] = pd.get_dummies(df['embarked'], dtype=int)
# transformando idade para inteiro
df['age'] = df[['age']].apply(numeric_age, axis=1)
df['age'] = pd.to_numeric(df['age'])
# aplicando para datas nulas a médias das idades de sua classe social e sexo
df['age'] = df[['age', 'sex', 'pclass']].apply(handle_age, axis=1)
# transformando o valor pago em float
df['fare'] =  df[['fare']].apply(handle_fare, axis=1)
df = clean_dataset(df)
""" Separando dados de treino e teste """
x = df.drop('survived', axis=1)
y = df['survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
""" Construcão dos modelos """
models = {
    'logistic_regression': LogisticRegression(
        penalty='l2',
        max_iter=1000
    ),
    'tree': DecisionTreeClassifier(
        criterion='gini',
        splitter='best'
    ),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(
        n_estimators=1000,
        criterion='gini',
    ),
    'svm': SVC()
}
results = {}
""" Predicões """
for index, (name, model) in enumerate(models.items()):
    model.fit(x_train, y_train)
    predicts = model.predict(x_test)
    results[name] = {
        'Accuracy Score': accuracy_score(y_test, predicts),
        'Confusion Matrix': confusion_matrix(y_test, predicts),
        'Classification Report': classification_report(y_test, predicts),
    }
    print(results[name])
    # save model
    dump(model, f"{name}.joblib")