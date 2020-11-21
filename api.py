#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import IntEnum
from typing import Generator

import uvicorn
from fastapi import Depends, FastAPI
from joblib import load
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

app = FastAPI(title='API - Titanic Survivor')


class EconomicClassEnum(IntEnum):
    low = 1
    middle = 2
    high = 3


class SexEnum(IntEnum):
    male = 0
    female = 1


class SurvivorRequestSchema(BaseModel):
    economic_class: EconomicClassEnum
    age: int
    sex: SexEnum
    number_of_siblings: int
    number_of_parents: int
    ticket_value: float
    is_embarked: bool

    def get_values(self):
        return [
            self.economic_class,
            self.sex,
            self.age,
            self.number_of_siblings,
            self.number_of_parents,
            self.ticket_value,
            self.is_embarked
        ]


class SurvivorResponseSchema(BaseModel):
    survived: bool
    probability_of_living: float
    probability_of_dying: float


def get_logistic_regression() -> Generator:
    try:
        model = load('logistic_regression.joblib')

        yield model
    finally:
        model = None


def get_tree() -> Generator:
    try:
        model = load('tree.joblib')

        yield model
    finally:
        model = None


def get_naive_bayes() -> Generator:
    try:
        model = load('naive_bayes.joblib')

        yield model
    finally:
        model = None


def get_svm() -> Generator:
    try:
        model = load('svm.joblib')

        yield model
    finally:
        model = None


def get_random_forest() -> Generator:
    try:
        model = load('random_forest.joblib')

        yield model
    finally:
        model = None


@app.post("/logistic-regression")
def logistic_regression(
    schema: SurvivorRequestSchema,
    model: LogisticRegression = Depends(get_logistic_regression)
):
    predict = model.predict([schema.get_values()])[0]
    probability = model.predict_proba([schema.get_values()])[0]

    return SurvivorResponseSchema(
        survived=predict == 1,
        probability_of_dying=probability[0],
        probability_of_living=probability[1]
    )


@app.post("/tree")
def tree(
    schema: SurvivorRequestSchema,
    model: DecisionTreeClassifier = Depends(get_tree)
):
    predict = model.predict([schema.get_values()])[0]
    probability = model.predict_proba([schema.get_values()])[0]

    return SurvivorResponseSchema(
        survived=predict == 1,
        probability_of_dying=probability[0],
        probability_of_living=probability[1]
    )


@app.post("/naive-bayes")
def naive_bayes(
    schema: SurvivorRequestSchema,
    model: GaussianNB = Depends(get_naive_bayes)
):
    predict = model.predict([schema.get_values()])[0]
    probability = model.predict_proba([schema.get_values()])[0]

    return SurvivorResponseSchema(
        survived=predict == 1,
        probability_of_dying=probability[0],
        probability_of_living=probability[1]
    )


@app.post("/svm")
def svm(
    schema: SurvivorRequestSchema,
    model: SVC = Depends(get_svm)
):
    predict = model.predict([schema.get_values()])[0]
    probability = model.predict_proba([schema.get_values()])[0]

    return SurvivorResponseSchema(
        survived=predict == 1,
        probability_of_dying=probability[0],
        probability_of_living=probability[1]
    )


@app.post("/random-forest")
def random_forest(
    schema: SurvivorRequestSchema,
    model: RandomForestClassifier = Depends(get_random_forest)
):
    predict = model.predict([schema.get_values()])[0]
    probability = model.predict_proba([schema.get_values()])[0]

    return SurvivorResponseSchema(
        survived=predict == 1,
        probability_of_dying=probability[0],
        probability_of_living=probability[1]
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
