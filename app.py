import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input


app = Flask(__name__)

# Load the Alzheimer's prediction model
model = load_model(r"Alzheimer.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET","POST"])
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        
        img=image.load_img(filepath,target_size=(128,128,3))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        
        img_data=preprocess_input(x)
        prediction=np.argmax(model.predict(img_data), axis=1)
        
        index=['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        
        result=str(index[prediction[0]])
        print(result)
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
