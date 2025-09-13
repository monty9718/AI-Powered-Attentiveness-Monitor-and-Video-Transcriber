import whisper

# Load Whisper model
model = whisper.load_model("base")

# Path to your video file
video_path = "video.mp4"

# Generate transcript
result = model.transcribe(video_path)

# Save transcript to a text file
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("✅ Transcription completed! Saved to transcript.txt")

