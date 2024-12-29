from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import speech_recognition as sr
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
import librosa

app = Flask(__name__, static_folder='static')


# Load the trained model
model = load_model("stress_detector_cnn.h5")

# Function to preprocess audio for the model
def preprocess_audio(file_path):
    SAMPLE_RATE = 16000       # Ensure sample rate matches the model's training
    MAX_DURATION_SEC = 5.0    # Ensure maximum duration matches
    N_MFCC = 40               # Number of MFCC features
    FIXED_FRAMES = 200        # Fixed number of frames along the time axis

    try:
        # Load the audio file using librosa (up to MAX_DURATION_SEC seconds)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SEC)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

        # Pad or truncate to FIXED_FRAMES
        if mfcc.shape[1] < FIXED_FRAMES:
            pad_width = FIXED_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :FIXED_FRAMES]

        # Expand dimensions for compatibility with the model: (40, 200) -> (40, 200, 1)
        mfcc = np.expand_dims(mfcc, axis=-1)

        # Add an extra batch dimension for prediction: (40, 200, 1) -> (1, 40, 200, 1)
        return np.expand_dims(mfcc, axis=0)

    except Exception as e:
        print(f"Error in preprocess_audio: {e}")
        raise e


@app.route('/')
def home():
    return render_template('home.html')

@app.route("/signup")
def signup():
    name = request.args.get('username','')
    number = request.args.get('number','')
    email = request.args.get('email','')
    password = request.args.get('psw','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `detail` (`name`,`number`,`email`, `password`) VALUES (?, ?, ?, ?)",(name,number,email,password))
    con.commit()
    con.close()

    return render_template("signup-in.html")


@app.route("/signin")
def signin():

    mail1 = request.args.get('name','')
    password1 = request.args.get('psw','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `name`, `password` from detail where `name` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()
    print(data)

    if data == None:
        return render_template("signup-in.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index1.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index1.html")
    else:
        return render_template("signup-in.html")

@app.route('/signout')
def signout():
	return render_template('signup-in.html')

@app.route("/index1", methods=["GET", "POST"])
def index1():
    return render_template('index1.html')

@app.route("/predict1", methods=["GET", "POST"])
def predict1():
    transcript = ""
    #result = "No Prediction"
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        predicted_label = -1
        if file:
            # Save the file temporarily as a .wav file
            file_path = "temp_audio.wav"
            file.save(file_path)
            
            # Transcribe audio using SpeechRecognition
            recognizer = sr.Recognizer()
            
            try:
                with sr.AudioFile(file_path) as source:
                    data = recognizer.record(source)
                    transcript = recognizer.recognize_google(data, key=None)
                print("Transcript:", transcript)
            except Exception as e:
                print("Error during transcription:", e)
                transcript = "Could not process audio transcription."
            
            try:
                processed_audio = preprocess_audio(file_path)
                prediction = model.predict(processed_audio)
                predicted_label = np.argmax(prediction)
                print(predicted_label)
            except Exception as e:
                print("Error during Processing:", e)

        if predicted_label == 1:
            return render_template('index1.html',prediction = predicted_label,transcript=transcript)

        elif predicted_label == 2:
            return render_template('index1.html',prediction = predicted_label,transcript=transcript)

        else:
            return render_template('index1.html',prediction = predicted_label,transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
