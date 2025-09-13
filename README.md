# AI-Powered Attentiveness Monitor & Video Transcriber

A dual-function desktop application built with Python that uses computer vision to track user focus and an AI module to transcribe video files.

## About The Project

This tool was developed to address two common needs in online learning and remote work: **quantifying focus** and **making video content accessible**. The application has two core features:

1.  **Attentiveness Monitor:** Uses a webcam to detect a user's face and eyes in real-time. It intelligently calculates the total time the user is actively looking at the screen, providing a metric for focus sessions.
2.  **Video Transcriber:** Allows a user to upload a video file. It then processes the audio and generates a full, accurate text transcript, which can be saved for later use.

This project combines the power of **Computer Vision** for engagement analysis and **Speech-to-Text** technology for content transcription into one user-friendly tool.

## Key Features

* **Real-Time Focus Tracking:** Monitors user engagement live via webcam.
* **Attention Analytics:** Quantifies and displays the total time a user was attentive.
* **Automated Transcription:** Upload a video file to generate a text transcript automatically.
* **Exportable Content:** Save the generated transcripts as text files.
* **Simple UI:** A clean and intuitive interface built for ease of use.

## Built With

* Python
* OpenCV
* Dlib
* A Speech-to-Text Library
* A Python GUI Framework

## Getting Started

To get a local copy up and running, follow these simple steps.

### Installation

1.  Clone this repository to your local machine.
2.  Navigate to the project directory.
3.  Install the required packages from the `requirements.txt` file.
    ```sh

### Usage

1.  Run the main application file.
    ```sh
    python main.py
    ```
2.  Follow the on-screen instructions to either start the camera for monitoring or upload a video for transcription.
