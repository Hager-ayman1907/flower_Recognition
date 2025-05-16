import os
#import keras
from tensorflow import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.header('🌹Flower Classification CNN Model🌹')
## creating a function for prediction
def animal_prediction(input_data):
    # دمج البيانات في نص واحد
    new_combined = ' '.join(input_data)
    new_transformed = vectorizer.transform([new_combined])

    # التأكد من أن عدد الميزات متطابق
    if new_transformed.shape[1] != loaded_model.n_features_in_:
        raise ValueError(f"Expected {loaded_model.n_features_in_} features, but got {new_transformed.shape[1]}")

    # التنبؤ بالفئة
    prediction = loaded_model.predict(new_transformed)
    return prediction


# Function to render the initial cube animation with 6 cubes
def render_js_animation():
    animation_code = """
    <div class="container" style="position:relative; height:300px; display:flex; justify-content:center; align-items:center;">
        <div class="flex">
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
        </div>
        <div class="flex">
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
        </div>
    </div>
    <style>
        .flex {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }

        .cube {
            position: relative;
            width: 60px;
            height: 60px;
            transform-style: preserve-3d;
            animation: rotate 4s infinite;
            margin: 10px;
        }

        .wall {
            position: absolute;
            width: 60px;
            height: 60px;
            background: rgba(255, 255, 255, 0);
            border: 1px solid #000000 ;
        }

        .front { transform: translateZ(30px); }
        .back { transform: translateZ(-30px) rotateY(180deg); }
        .left { transform: rotateY(-90deg) translateX(-30px); transform-origin: center left; }
        .right { transform: rotateY(90deg) translateX(30px); transform-origin: center right; }
        .top { transform: rotateX(90deg) translateY(-30px); transform-origin: top center; }
        .bottom { transform: rotateX(-90deg) translateY(30px); }

        @keyframes rotate {
            0% { transform: rotateX(0deg) rotateY(0deg); }
            50% { transform: rotateX(180deg) rotateY(180deg); }
            100% { transform: rotateX(360deg) rotateY(360deg); }
        }
    </style>
    """
    st.markdown(animation_code, unsafe_allow_html=True)


# Function to render result animation
def render_result_animation(result):
    if result == 1:
        animation_code = """
        <div style="display:flex;justify-content:center;align-items:center;height:200px;">
            <div style="width:100px;height:100px;border-radius:50%;background-color:#ff5722;animation:bounce 1s infinite;"></div>
            <style>
                @keyframes bounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-20px); }
                }
            </style>
        </div>
        """
    else:
        animation_code = """
        <div style="display:flex;justify-content:center;align-items:center;height:200px;">
            <div style="width:100px;height:100px;border-radius:50%;background-color:#4caf50;animation:scale 1s infinite;"></div>
            <style>
                @keyframes scale {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.2); }
                }
            </style>
        </div>
        """
    st.markdown(animation_code, unsafe_allow_html=True)
render_js_animation()

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('Flower_Recog_Model.h5')

# style change
st.markdown(
"""
<style>
.stApp {
    background-color: #FFB6B9;  
    color: #333333;
}
</style>
""",
 unsafe_allow_html=True
)

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    os.makedirs('upload', exist_ok=True)  # ← يتأكد إن المجلد موجود
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width=200)

    st.markdown(classify_images(uploaded_file))

