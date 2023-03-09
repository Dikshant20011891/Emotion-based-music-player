from flask import Flask,render_template, Response, redirect, url_for, request
import os
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
import os

app = Flask(__name__)

path = 'model'
label = ""
face_classifier = cv2.CascadeClassifier(r'static/model/haarcascade_frontalface_default.xml')
classifier = load_model(r'static/model/my_model.h5')
emotion_labels =  { 0: "angry", 1: "happy", 2: "neutral",3: "sad"}

#define video capture object
cap = cv2.VideoCapture(0)

def gen_frames():
    global label

    while True:

        # capture the video frame by frame
        _, frame = cap.read()

        # return an image to gray scale
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(gray)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation = cv2.INTER_AREA)

            if np.sum([roi_gray] != 0):
                roi = roi_gray.astype('float')/255.0
                roi = tf.keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis = 0)

                prediction = list(classifier.predict(roi)[0])
                label = emotion_labels[prediction.index(max(prediction))]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        # convert image format to data stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/')
def home():
    global cap
    cap = cv2.VideoCapture(0)
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    # video streaming using flask
    return Response(gen_frames(),mimetype = 'multipart/x-mixed-replace; boundary=frame')

song_count = 0

@app.route('/songs', methods=['GET','POST'])
def play():
    cap.release()
    global song_count
    global label
    if request.method == 'POST':
        if "next" in request.form:
            song_count = song_count+1
        elif "previous" in request.form:
            song_count = song_count-1
        if "angry" in request.form:
            label = 'angry'
        elif "happy" in request.form:
            label = 'happy'
        elif "neutral" in request.form:
            label = 'neutral'
        elif "sad" in request.form:
            label = 'sad'
    files = []
    path = 'static/songs/'+label
    css_path = 'style/css/'+label+'.css'
    mysongs = os.listdir(path)
    for name in mysongs:
        files.append(name)
    if(song_count >= len(files) or song_count < 0):
        song_count = 0

    song_img = 'static/style/' + label +'_img/' + label + song_count.__str__() + '.png'
    curr_song = files[song_count]
    # print(curr_song)

    return render_template('player.html', song = curr_song, dir = path,song_img = song_img, emotion = css_path, type = label)

if __name__ == '__main__':
    app.run(debug = True)