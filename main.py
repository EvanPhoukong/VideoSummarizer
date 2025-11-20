import whisper
from pathlib import Path
import nltk

class VideoSummarizer:
    """
    Give an input video, the VideoSummarzier will be able to generate an abstract summary
    that highlights key concepts and ideas in the video.
    """

    def __init__(self, video: str) -> None:
        self.video = video


    def run(self) -> None:
        transcript = self.get_transcript()
        

    def get_transcript(self) -> str:
        """
        Use whisper to transcribe the input video
        """

        model = whisper.load_model("turbo")
        transcript = model.transcribe(self.video)["text"]
        # nltk.download('punkt_tab')
        # transcript = nltk.sent_tokenize(transcript)

        with open('transcript.txt', 'w+') as file:
            # for line in transcript:
            #     file.write(line + '\n')
            file.write(transcript)

        return transcript


    def text_preprocessing(self):
        """
        Text is normalized using standard preprocessing procedure, incluidng stemming, lemmatization, POS, etc.
        """

    def convert_text_to_embeddings(self):
        """
        Covert the text into numerical vectors/embeddings that capture meaninng and context
        """

    def summarize(self):
        """
        Summarize text using ChatGPT API
        """

    def evaluation(self):
        """
        Evaluate the performance of the model
        """


def main() -> None:

    video = r"C:\Users\Evan\Desktop\Lab12\payload.mp4"
    model = VideoSummarizer(video)
    model.run()



if __name__ == "__main__":
    main()