import numpy as np
import wave
import tensorflow as tf
import tempfile
import os
import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)
model = tf.keras.models.load_model('model')


def create_melspectrogram(file_path, sr=22050, n_mels=128, fmax=8000, duration=None):
    # 음성 파일 로드
    y, sr = librosa.load(file_path, sr=sr)
    # duration이 주어진 경우, 해당 시간만큼의 샘플 수를 계산
    if duration:
        max_length = int(sr * duration)
        y = librosa.util.fix_length(y, max_length)
    # 멜 스펙토그램 생성
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    # 로그 스케일로 변환
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def process_audio_files(file, duration=None):
    S_dB = create_melspectrogram(file, duration=5)
    return S_dB

@app.route('/')
def home():
   return 'This is Home!'


@app.route('/classify-emotion', methods=['POST'])
def classify_emotion():

   if 'file' not in request.files:
        return 'File is missing', 404
   voice = request.files['file']

   filePath = os.path.join(tempfile.gettempdir(), voice.filename)
   voice.save(filePath)

   tensor = create_melspectrogram(filePath, duration=5)
   
   tensor = tensor[np.newaxis, ..., np.newaxis]
   

   result = model.predict(tensor)

   response = {
      'class': int(np.argmax(result[0]))
   }
   return jsonify(response) 

if __name__ == '__main__':  
   app.run('0.0.0.0',port=8080,debug=True)