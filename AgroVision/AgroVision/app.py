import random
import numpy as np
import pickle
import json
from flask import jsonify
import os
import requests

from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from googletrans import Translator
from gevent.pywsgi import WSGIServer
from tensorflow.keras.preprocessing import image

from nltk.stem import WordNetLemmatizer

import webview 
lemmatizer = WordNetLemmatizer()

# chat initialization
model = load_model("chatbot_modelz.h5")
intents = json.loads(open("intentsz.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))



class_dict = {0:'Diseased cotton leaf',
              1:'Diseased cotton plant',
              2:'Fresh cotton leaf',
              3:'Fresh cotton plant' }

# Load your trained model
model_plant = load_model("incep.h5")

# initialize translator

talk = Translator()

#MODEL_PATH ='model_resnet50.h5'

# Load your trained model
#model = load_model(MODEL_PATH)



app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/news")
def news():
    return render_template("news.html")

@app.route("/newsDetails")
def newsdetails():
    return render_template("news-detail.html")

@app.route("/donate")
def donate():
    return render_template("donate.html")


@app.route("/chatbot")
def chatbot():
    
    return render_template("chatbot.html")

@app.route("/disease")
def plant_disease():
    
    return render_template("diseasePredict.html")

@app.route("/crops")
def crops():
    return render_template("crops.html")


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route("/climaticPlant")
def climaticplant():
    return render_template("climatic_plants.html")

@app.route("/climaticPlant/<id>")
def detailClimate(id):
    if id=="Tropical":
        return render_template("details_climatic.html",id=id)
    elif id=="Sub-Tropical":
        return render_template("details_climatic.html",id=id)
    elif id=="Temperature":
        return render_template("details_climatic.html",id=id)
    elif id=="Cool-Season":
        return render_template("details_climatic.html",id=id)
    elif id=="Warm-Season":
        return render_template("details_climatic.html",id=id)
    else:
        return render_template("details_climatic.html")
    

@app.route("/crop/<id>")
def tomato(id):
    if id=="tomato":
        return render_template("tomato.html",id=id)
    elif id=="cotton":
        return render_template("tomato.html",id=id)

 

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    lang = request.form.get("language")
    
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        #out = talk.translate(res, dest="mr")
        #out = out.text
        #print(out)
        #if lang=='eng':
        return res
        #else:
            #return out

    


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    
    return str(result)



def model_predict(model,img_path ):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
   
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model_plant.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
        
    return preds


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(model_plant,file_path)
        result = preds
        return result
    return None 







@app.route('/weather', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        city_name = request.form.get('city')

        #take a variable to show the json data
        r = requests.get('https://api.openweathermap.org/data/2.5/weather?q='+city_name+'&appid=ba3b83ce3a099ae671bb31267c01c70e')

        #read the json object
        json_object = r.json()

        #take some attributes like temperature,humidity,pressure of this 
        temperature = int(json_object['main']['temp']-273.15) #this temparetuure in kelvin
        humidity = int(json_object['main']['humidity'])
        pressure = int(json_object['main']['pressure'])
        wind = int(json_object['wind']['speed'])

        #atlast just pass the variables
        condition = json_object['weather'][0]['main']
        desc = json_object['weather'][0]['description']
        
        return render_template('weather.html',temperature=temperature,pressure=pressure,humidity=humidity,city_name=city_name,condition=condition,wind=wind,desc=desc)
    else:
        return render_template("weather.html")
	
	
if __name__ == '__main__':
    app.run(debug=True)
    #webview.start()