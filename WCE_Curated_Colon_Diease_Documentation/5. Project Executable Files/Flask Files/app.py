import numpy as np
import os
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow.keras.utils import load_img, img_to_array
import pickle


app=Flask(__name__)

model = pickle.load(open('face.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/output',methods=['GET','POST'])
def output():
    if request.method == 'POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=load_img(filepath,target_size=(224,224))
        image_array=np.array(img)
        image_array=np.expand_dims(image_array,axis=0)
        pred=np.argmax(model.predict(image_array),axis=1)
        index=['Normal','Ulcerative','Polyps','Esophagitis']
        prediction=index[int(pred)]
        print("prediction")
        return render_template("output.html",predict=prediction)


if __name__ == '__main__':
    app.run(debug=True)