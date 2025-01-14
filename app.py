# English Learning App (Backend with ChatGPT API + Whisper Integration)
# Technologies: Flask (Python), OpenAI API (ChatGPT), Whisper (Speech-to-Text), Google TTS

import time
from flask import Flask, request, jsonify, send_file
import openai
import whisper
import imageio_ffmpeg as ffmpeg
from gtts import gTTS
import os
import tempfile
from flask_cors import CORS
from dotenv import load_dotenv
import jwt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
import firebase_admin
from firebase_admin import credentials, auth
from werkzeug.security import generate_password_hash


# Charger les informations d'authentification Firebase
cred = credentials.Certificate('path/to/your/firebase-adminsdk.json')  # Remplacez par le chemin de votre fichier JSON
firebase_admin.initialize_app(cred)

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load Whisper model (base for faster performance)
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# OpenAI API key configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# Override Whisper's internal ffmpeg path using imageio_ffmpeg
whisper.audio.ffmpeg = ffmpeg.get_ffmpeg_exe()

contexte = "You are an English learning assistant with a focus on personalized language acquisition. Your role is to provide clear, accurate, and contextually appropriate responses tailored to the learner's proficiency level (beginner, intermediate, advanced). Your guidance should cover the following areas: Grammar: You explain English grammar rules and provide practical examples.Pronunciation: You offer feedback and suggestions to improve pronunciation, focusing on common challenges for non-native speakers. Vocabulary: You suggest vocabulary based on the user's interests, context, and desired learning goals.Listening and Comprehension: You help improve listening skills through tailored exercises or recommendations. Conversational Skills: You engage in realistic dialogues, allowing the learner to practice everyday English in different contexts.Writing Skills: You provide writing corrections and offer constructive feedback on sentence structure, style, and vocabulary usage. Cultural Context: You explain idiomatic expressions, slang, and cultural references to help the learner understand the nuances of the language. Adaptability: You adjust your difficulty level based on the learner’s progress, giving progressively more complex sentences and challenges. You must only respond in English, and only in English. If the user communicates in another language, politely ask them to continue in English. Additionally, suggest a relevant topic of discussion based on the current context to encourage further conversation in English"

# Secret key for encoding the JWT (make sure to store it securely)
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY")  # Get from environment variable
jwt = JWTManager(app)

# Endpoint for speech-to-text conversion
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    file = request.files['file']
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    file.save(audio_path)

    # Transcribe audio using Whisper with updated torch.load security
    result = whisper_model.transcribe(audio_path, fp16=False)
    os.remove(audio_path)

    return jsonify({"transcription": result['text']})

# Endpoint for ChatGPT response
@app.route('/chat', methods=['POST'])
def chat_with_gpt():
    data = request.json
    user_input = data.get("message")
    conversation_history = data.get("history", [])

    # Send request to ChatGPT API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": contexte}] + conversation_history + [{"role": "user", "content": user_input}]
    )

    gpt_response = response['choices'][0]['message']['content']
    
    # Convert GPT response to audio using gTTS
    tts = gTTS(text=gpt_response, lang='en')
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(audio_path)

    # Add a small delay before attempting to delete the file
    time.sleep(1)  # Wait for the file to be fully processed

    return jsonify({"response": gpt_response, "audio": audio_path})

# Endpoint to fetch TTS audio
@app.route('/audio/<path:filename>', methods=['GET'])
def get_audio(filename):
    return send_file(filename, mimetype='audio/mpeg')

# Start the Flask app
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
