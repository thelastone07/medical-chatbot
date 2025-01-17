import joblib
import numpy as np
import pandas as pd

def get_disease(symptoms : list):
    model_data = joblib.load('./disease_model_data')

    model = model_data['model']
    selector = model_data['selector']
    vocab = model_data['vocab']
    le = model_data['le']

    x = [0] * len(vocab)
    for s in symptoms:
        if s in vocab:
            idx = vocab.index(s)
            x[idx] = 1
    x = np.array(x)
    x = x.reshape(1,-1)
    x = selector.transform(x)

    y_pred = model.predict(x)

    y = le.inverse_transform(y_pred)

    return y

def get_description(disease):
    descriptions = pd.read_csv('disease_Description.csv')
    desc = descriptions[descriptions['Disease'] == disease]
    desc = desc['Description'].tolist()[0]
    return desc

def get_precautions(disease):
    precautions = pd.read_csv('disease_precaution.csv')
    precs = precautions[precautions['Disease'] == disease].iloc[:,1:]
    precs_cols =  precs.columns
    prec = []
    for col in precs_cols:
        if precs[col].any():
            prec.append(precs[col].tolist()[0])

    return prec


symptomps1 = ['itching', 'acidity', 'fatigue']
disease = get_disease(symptomps1)[0]
desc = get_description(disease)
precautions = get_precautions(disease)
print(desc)
print(precautions)



