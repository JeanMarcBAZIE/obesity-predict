from flask import Flask, request, url_for, redirect, render_template, jsonify
from  sklearn import *
from sklearn.pipeline import make_pipeline
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)

pipe=pickle.load(open('finalized_model.sav','rb'))

cols=['Gender','Age','Weight','family_history_with_overweight','SCC']

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	int_features=[x for x in request.form.values()]
	final=np.array(int_features)
	data_unseen=pd.DataFrame([final], columns=cols)
	prediction=pipe.predict(data_unseen)
	#prediction=int(prediction.Label[0])
	return render_template('home.html', pred='Le type d\'état d\'Obèsité correspondant : {}'.format(prediction))


if __name__ == '__main__':

    app.run(debug=True)
    