import pytest
import io
import os
from flask import Flask
import sys
sys.path.insert(0, '.')

from . import app

# Crée un dossier pour stocker les fichiers audio générés
os.makedirs("./tests/outputs", exist_ok=True)

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Full API test: Send audio and get a response with another audio file
def test_full_api_flow(client):
    # Step 1: Create a mock audio file
    audio_data = io.BytesIO(open("./tests/test.mp3", "rb").read())
    audio_data.name = "test.mp3"

    # Step 2: Send audio to /transcribe endpoint
    transcribe_response = client.post('/transcribe', content_type='multipart/form-data', data={'file': (audio_data, 'test.mp3')})

    assert transcribe_response.status_code == 200
    transcription = transcribe_response.json.get('transcription')
    assert transcription is not None

    # Step 3: Send transcription to /chat endpoint
    chat_response = client.post('/chat', json={"message": transcription, "history": []})

    assert chat_response.status_code == 200
    gpt_response = chat_response.json.get('response')
    audio_path = chat_response.json.get('audio')

    assert gpt_response is not None
    assert audio_path is not None

    # Step 4: Save the audio response in the tests/outputs folder
    audio_response = client.get(f'/audio/{audio_path}')
    
    # Check if audio is successfully fetched
    assert audio_response.status_code == 200
    assert audio_response.mimetype == 'audio/mpeg'

    output_path = "./tests/outputs/test_response.mp3"
    with open(output_path, "wb") as f:
        f.write(audio_response.data)

    # Check if the file was saved correctly
    assert os.path.exists(output_path)
