Overview:
This is a prototype accent recognition application. It records or uploads voice samples and compared them to the stored accents (Irish, Scottish, Yorkshire). The application then shows the most likely accent and a confidence percentage alongside it. The system is built to demonstrate how machine learning features can be useful for voice analysis.

Features:
- Records voice directly in the app or you can upload a WAV file
- Displays the most likely accent match with a confidence percentage
- Shows a pie chart distribution of similarity across the accents
- Keeps a history of the last 20 recordings, which can be viewed in the app
- The recordings are processed but then discarded and only the results are logged for privacy reasons

How to run:

1. Clone the repository:
git clone https://github.com/kim210603/accent-recognition
cd accent-recognition

2. Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3. Install dependencies:
# I have frozen the environment for this
pip install -r ar_requirements.txt

4. Run the application:
streamlit run ar_app.py

Limitations:
- Only a few accents are included
- The app is based on limited data due to the use of TTS voices (ElevenLabs)