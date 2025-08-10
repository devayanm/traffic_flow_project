#!/bin/bash
set -e
python3 -m venv venv || true
. venv/bin/activate
pip install -r requirements.txt
python3 main.py --download_youtube --youtube_url "https://www.youtube.com/watch?v=MNn9qKG2UFI"
