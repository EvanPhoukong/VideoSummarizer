import whisper
from pathlib import Path


def get_transcript(video: str) -> None:
    """
    Use whisper to transcribe the input video
    """

    model = whisper.load_model("turbo")
    transcript = model.transcribe(video)["text"]

    with open('transcript.txt', 'w+') as file:
        file.write(transcript)


def main() -> None:

    video = r"C:\Users\Evan\Desktop\Lab12\payload.mp4"
    get_transcript(video)


if __name__ == "__main__":
    main()