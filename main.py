import whisper
from pathlib import Path
import nltk
import string, re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec

class VideoSummarizer:
    """
    Give an input video, the VideoSummarzier will be able to generate an abstract summary
    that highlights key concepts and ideas in the video.
    """

    def __init__(self, video: str) -> None:
        self.video = video


    def run(self) -> None:
        transcript = self.get_transcript()
        tokens = self.text_preprocessing(transcript)
        self.convert_text_to_embeddings(tokens)
        

    def get_transcript(self) -> str:
        """
        Use whisper to transcribe the input video
        """

        model = whisper.load_model("turbo")
        transcript = model.transcribe(self.video)["text"]
        # nltk.download('punkt_tab')
        # transcript = nltk.sent_tokenize(transcript)

        # with open('transcript.txt', 'w+', encoding='utf-8') as file:
        #     for line in transcript:
        #         file.write(line + '\n')
        #     file.write(transcript)

        return transcript


    def text_preprocessing(self, transcript: str):
        """
        Text is normalized using standard preprocessing procedure, incluidng stemming, lemmatization, POS, etc.
        """

        #Lower case everything
        transcript = transcript.lower()

        #Remove punctuation
       # transcript = transcript.translate(str.maketrans('', '', string.punctuation))
        tokens = transcript.split()

        #Remove stopwords
        nltk.download('stopwords')
        STOPWORDS = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in STOPWORDS]

        #Stem words
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        #Lemmatize words
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        transcript = ' '.join(tokens)

        #Tokenize the sentences
        transcript = nltk.sent_tokenize(transcript)

        #Remove punctuation from sentences
        transcript = [sent.translate(str.maketrans('', '', string.punctuation)) for sent in transcript]

        #Tokenize the sentences
        tokens = [nltk.word_tokenize(sent) for sent in transcript]

        print(transcript)
        print(tokens)

        return tokens


    def convert_text_to_embeddings(self, tokens):
        """
        Convert the text into numerical vectors/embeddings that capture meaninng and context
        """
        model = Word2Vec(sentences=tokens, min_count=1)
        vec = model.wv.key_to_index
        print(vec)

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