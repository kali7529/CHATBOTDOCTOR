import os
import time
import tempfile
import asyncio
from flask import Flask, request, jsonify, send_file, render_template, Response
from flask_cors import CORS
from dotenv import load_dotenv

# AI & Audio Libs
import google.generativeai as genai
import edge_tts
import speech_recognition as sr
from pydub import AudioSegment

# Load API Key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") # Ensure this is in your .env file
genai.configure(api_key=API_KEY)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# --- DOCTOR CONFIGURATION ---
DOCTOR_SYSTEM_PROMPT = """
You are Dr. Nova, a highly experienced, empathetic, and professional AI medical consultant. 
Your goal is to assist users with medical questions, symptom analysis, and health advice.

Rules:
1. Tone: Professional, reassuring, clinical but accessible (like a top-tier human doctor).
2. Knowledge: Use advanced medical knowledge (pharmacology, pathology, anatomy).
3. Structure: 
   - Acknowledge the symptoms.
   - Ask clarifying questions if needed.
   - Provide a differential diagnosis (potential causes).
   - Recommend treatments, home remedies, or medicines (generic names + common brands).
   - ALWAYS conclude with a disclaimer: "I am an AI. Please consult a physical doctor for emergencies."
4. Formatting: Use Markdown. Use **Bold** for medicine names and headers. Use lists for symptoms/treatments.
"""

# Store chat history in memory (for demo purposes)
chat_history = {}

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json
    user_msg = data.get("message", "")
    chat_id = data.get("chat_id", "default")

    # Initialize history if new
    if chat_id not in chat_history:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=DOCTOR_SYSTEM_PROMPT
        )
        chat_history[chat_id] = model.start_chat(history=[])
    
    chat = chat_history[chat_id]

    def generate():
        try:
            response = chat.send_message(user_msg, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n**Error:** {str(e)}"

    return Response(generate(), mimetype="text/plain")

@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "")
    # Use a calm, professional voice
    voice = "en-US-BrianNeural" 
    
    if not text: return jsonify({"error": "No text"}), 400

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out_file.close()

    async def run_tts():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_file.name)

    asyncio.run(run_tts())
    return send_file(out_file.name, mimetype="audio/mpeg")

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"text": ""})

    file = request.files["audio"]
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        temp_webm.write(file.read())
        webm_path = temp_webm.name

    wav_path = webm_path + ".wav"

    try:
        # Convert WebM to WAV (Requires FFmpeg)
        AudioSegment.from_file(webm_path).export(wav_path, format="wav")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            # Recognize
            text = recognizer.recognize_google(audio_data)
            return jsonify({"text": text})
    except Exception as e:
        print(f"STT Error: {e}")
        return jsonify({"text": "", "error": str(e)})
    finally:
        # Cleanup
        if os.path.exists(webm_path): os.remove(webm_path)
        if os.path.exists(wav_path): os.remove(wav_path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)