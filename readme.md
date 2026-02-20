# Project Name
Exctract video clips from master
## Description
An automated video editing tool that uses artificial intelligence to find specific voices across multiple long-form videos. Simply provide one reference sample, and the tool will scan a folder of videos, identify every time that person speaks, and extract those moments into high-quality, Windows-compatible clips.
## Features

## Tech Stack
Python 3.12
FFmpeg (Must be added to your system's Environment Variables) 

## Installation
```bash
cd project
python -m venv venexv
venexv\Scripts\activate
pip install -r requirements.txt

pip freeze > requirements.txt
pip install demucs resemblyzer numpy soundfile librosa