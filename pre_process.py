import os
import json
import subprocess
import whisper
from PIL import Image
import pytesseract


def convert_to_audio(video_path: str, output_audio: str = "lecture_audio.mp3") -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", output_audio],
        check=True
    )
    return output_audio


def create_transcript(audio_path: str, model_size: str = "tiny") -> list:
    """
    Generate transcript with timestamps using Whisper.
    Returns list of dicts: [{"start": float, "end": float, "text": str}, ...]
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)

    transcript = []
    for seg in result["segments"]:
        transcript.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    return transcript


def extract_slide_at(video_path: str, timestamp: int, slide_path: str):
    """
    Extract a single slide image from video at given timestamp.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path, "-frames:v", "1", slide_path],
        check=True
    )


def ocr_slide(slide_path: str) -> str:
    """
    Extract text from a slide image using Tesseract OCR.
    """
    try:
        return pytesseract.image_to_string(Image.open(slide_path))
    except Exception:
        return ""


def create_lecture_json(video_path: str, interval: int = 20,
                        model_size: str = "tiny",
                        output_json: str = "lecture.json") -> str:
    """
    Create a chunk-level JSON mapping:
    [
      { "start": ..., "end": ..., "transcript": "...", "slide_image": "...", "slide_text": "..." },
      ...
    ]
    """

    # --- Audio + Transcript ---
    audio_path = convert_to_audio(video_path)
    transcript_segments = create_transcript(audio_path, model_size)

    # --- Get video duration ---
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True,
        text=True
    )
    duration = float(result.stdout.strip())

    os.makedirs("slides", exist_ok=True)

    chunks = []
    t = 0
    slide_id = 1
    while t < duration:
        # --- Slide ---
        slide_path = os.path.join("slides", f"slide_{slide_id:03d}.jpg")
        extract_slide_at(video_path, t, slide_path)
        slide_text = ocr_slide(slide_path)

        # --- Transcript portion in this interval ---
        transcript_chunk = [
            seg["text"] for seg in transcript_segments
            if seg["start"] >= t and seg["end"] < t + interval
        ]
        chunk_text = " ".join(transcript_chunk).strip()

        chunks.append({
            "start": t,
            "end": min(t + interval, duration),
            "transcript": chunk_text,
            "slide_image": slide_path,
            "slide_text": slide_text
        })

        t += interval
        slide_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    return output_json


if __name__ == "__main__":
    video_file = "lecture.mp4"
    json_file = create_lecture_json(video_file, interval=20, model_size="base")
    print(f"Lecture data saved in {json_file}")
