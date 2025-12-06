import whisper
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
from openai import OpenAI
import os, sys
from bert_score import score
from summac.model_summac import SummaCConv
from numpy import ndarray, float64
from tkinter import filedialog
from tkinter import Tk
from yt_dlp import YoutubeDL
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

#Specify paths for transcript and generated summary files
file_path = os.path.dirname(os.path.realpath(__file__))
transcript_file = os.path.join(file_path, 'transcript.txt')
gen_sum = os.path.join(file_path, "generated.txt")

#Store the reference summaries in a dictionary
ref_sums = {}

#Store the video names in a dictionary
vids = {0 : r"Binary_Logical_Shifts.mp3", 
        1: r"LANs_and_WANs.mp3",
        2: r"Types_of_PL.mp3"}

class VideoSummarizer:
    """
    Give an input video, the VideoSummarzier will be able to generate an abstract summary
    that highlights key concepts and ideas in the video.
    """

    def __init__(self, video: str, selection: str) -> None:
        """
        Initialize the class, and store the selected video
        """
        self.video = video
        self.selection = selection


    def run(self) -> None:
        """
        Run the NLP Pipeline
        """

        #Step 1: Generate Transcript
        print("Generating Transcript...")
        transcript = self.get_transcript()

        #Step 2: Preprocess transcript and covert it into tokens
        print("Preprocessing Text...")
        tokens = self.text_preprocessing(transcript)

        #Step 3: Generate embeddings from tokens
        print("Generating Embeddings...")
        embeddings = self.convert_text_to_embeddings(tokens)

        #Step 4: Summarize the video using the transcript and embeddings
        print("Generating Summary...")
        self.summarize(embeddings, transcript)

        #step 5: Evaluate the performance of the model using BERTScore and SummaC
        if self.selection != 3:
            print("Evaluating model...")
            self.evaluation()


    def generate_summary(self, prompt: str) -> str:
        """
        Query the OpenAI API and generate a summary
        """

        #Initialize OpenAI Client
        client = OpenAI()
        
        #Query the client and retrieve the response
        try:
            response = client.responses.create(
                model="gpt-5-nano",
                input=prompt
            )
        except: #API Key not configured properly
            print('\nSUMMARY COULD NOT BE GENERATED')
            configure_API_key_instr()
            sys.exit()
        
        return response.output_text


    def get_transcript(self) -> str:
        """
        Use whisper to transcribe the input video
        """

        #Initialize Whisper (Turbo model) and transcribe video
        model = whisper.load_model("turbo")
        transcript = model.transcribe(self.video)["text"]

        #Tokenize the transcript into sentences
        sentences = nltk.sent_tokenize(transcript)
        with open(transcript_file, 'w+', encoding='utf-8', errors='ignore') as file:

            #Store each sentence as line in transcript file
            for s in sentences:
                file.write(s + '\n')

        return transcript


    def text_preprocessing(self, transcript: str):
        """
        Text is normalized using standard preprocessing procedure, including stemming, lemmatization, POS, etc.
        """

        #Lower case everything
        transcript = transcript.lower()

        #Conver transcript into tokens
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

        #Convert to string
        transcript = ' '.join(tokens)

        #Tokenize the sentences
        transcript = nltk.sent_tokenize(transcript)

        #Remove punctuation from sentences
        transcript = [sent.translate(str.maketrans('', '', string.punctuation)) for sent in transcript]

        #Tokenize the sentences
        tokens = [nltk.word_tokenize(sent) for sent in transcript]

        #print(transcript)
        #print(tokens)

        return tokens


    def convert_text_to_embeddings(self, tokens: list[list[str]]) -> ndarray[float64]:
        """
        Convert the text into numerical vectors/embeddings that capture meaninng and context
        """

        #Use Word2Vec to convert tokens into embeddings
        model = Word2Vec(sentences=tokens, min_count=1)
        vec = model.wv.vectors

        #print(vec)
        return vec
    

    def summarize(self, embeddings, transcript: str) -> None:
        """
        Summarize text using ChatGPT API
        """

        #Initialize Prompt 
        prompt = f"""
                You are given a lecture transcript and a set of supporting numerical features 
                (embeddings derived from the transcript). Use the transcript as the primary source of truth, 
                and use the numerical features only to reinforce recurring themes, detect important 
                topics, or identify patterns that should be included in the summary.

                Do NOT mention the numerical features, vectors, embeddings, models, 
                pipelines, or analysis methods in your output. 
                The final summary must read naturally, as if created purely from the lecture.

                Transcript:
                {transcript}

                Supporting features / Embeddings:
                {embeddings}

                Produce a structured-notes style summary that includes:
                - the key concepts and insights
                - important definitions or formulas
                - relationships between ideas
                - examples mentioned in the lecture
                - any emphasized themes reflected across the transcript

                Write the summary cleanly as lecture notes. Do not reference the existence 
                of supporting features, analysis methods, or summarization processes.
                """
        
        response = self.generate_summary(prompt)

        print('\n' + response)

        #Write generated summary to file
        with open(gen_sum, 'w+', encoding='utf-8', errors='ignore') as file:
            file.write(response)


    def evaluation(self) -> None:
        """
        Evaluate the performance of the model using BERTScore and SummaC
        BERTScore measures precision, recall, and F1 scores
        SummaC (Summary Consistency) measures factual consistency and validity
        """

        #Reference summary for lecture video on binary logical shifts
        ref_sums[0] = """
        Binary logical shifts move bits left to rights. Gaps created by the shift at the beginning or end are filled with 0s.
        For a left logical shift, we are multiplying the number by two each time we perform a shift. Due to size limits,
        we can lose data if we shift out of allocated memory, also known as an overflow error. For a right logical shift, 
        we are dividing by two each shift and lose the fractional part.
        """

        #Reference summary for lecture video on LANs and WANs
        ref_sums[1] = """
        LAN stands for local area network, and it exists over a small geographical area. 
        All infrastructure, hardware and software, is usually owned by manager of network. 
        WANs stand for wide area networks, and it exists over a larger geographical area, often multiple buildings. 
        A WAN contains infrastructure owned by multiple people or organizations. In practice, 
        a company with offices across different locations are connected through the wide area network. 
        One example of a WAN is the internet, since its massive and involves a lot of shared infrastructure.
        """

        #Reference summary for lecture video on programming languages
        ref_sums[2] = """
        A programming language is a set of words, symbols, and syntax that are used to write instructions a 
        computer can execute. There are two types of programming languages: High Level and Low Level programming 
        languages. An example of high level includes Python, OCR, and JavaScript. An example of low level includes 
        assembly or LMC, object code, and machine code. The CPU can only execute machine code and everything else 
        has to be translated to that. Low level languages have a low level of abstraction and do not simplify much 
        for the programmer. This can make them harder to program and maintain. They do allow a programmer to directly 
        control the hardware, manipulate memory, specify addresses and CPU operations, etc. As a result, program can 
        be made to run fast and take up little memory, which is important for simple systems like embedded systems. 
        Low level languages are not portable, and they only work on the CPU they were designed for, since different 
        machines have different machine codes. High level languages have a higher level of abstractions, with one 
        line of code equating to multiple low-level lines. A programmer doesn't need to understand how the hardware 
        works. More written English makes the more readable, and built-in subroutines make it quicker to program and 
        maintain code. High level programs are less efficient, taking up more memory and running slower. They are 
        portable, so they can run on any computer as long as a translate exists to convert it into the computer's 
        machine code.
        """

        #Obtain the reference summary
        reference = ref_sums[self.selection]

        #Open and stored the generated summary as a string
        with open(gen_sum, 'r', encoding='utf-8', errors="ignore") as file:
            generated = file.read()

        #--------------------------------------------------------------------------------------------------
        #NOTE: The below lines of code are a rough implementation of prompting the API to remov headers
        #and formattings from the user's generated summary, since these negatively impact SummaC Scores.

        # prompt = f"""Delete the title of the sumamry and any headers. Headers are almost always the first line of 
        #             a new section, with sections being seperated by empty lines. Convert mathematical symbols/operations
        #             into their English counterpart, such as + to plus.\n {generated}
        #             """

        # generated = self.generate_summary(prompt)
        # print('\n' + generated)
        #--------------------------------------------------------------------------------------------------

        #Calculate BertScore - Precision, Recall, F1
        P, R, F1 = score([generated], [reference], lang="en", verbose=True)

        #Initialize summary consistency model
        model = SummaCConv()

        #Open and store the transcript as a string
        with open(transcript_file, 'r', encoding='utf-8', errors="ignore") as file:
            transcript = file.read()

        #Calculate and print the raw score, in decimal form
        score1 = model.score([transcript], [generated])
        #print(score1)

        #Format the score to appear as a percentage
        score1 = float(score1['scores'][0])

        #Print metrics to terminal
        print("\nBERTScore")
        print(f"Precision: {P.mean().item() * 100: .2f}%")
        print(f"Recall: {R.mean().item() * 100: .2f}%")
        print(f"F1: {F1.mean().item() * 100: .2f}%\n")
        print(f'SummaC Score: {score1 * 100: .2f}%')


def retrieve_video() -> None:
    """
    Retrieve the video from user input, which can be a youtube URL, a downloaded video, or a preconfigured video
    """

    #Select what video you would like to summarize
    selection = input("Please enter the number of the video you would like to analyze: [0] Bit shifts, [1] LANs and WANs, [2] Programming Languages.\n" \
                            "If you would like to summarize your own video, please enter 3: ")
    
    #Handle incorrect inputs
    while selection not in ['0', '1', '2', '3']:
        selection = input("Please enter a valid number corresponding to the video you would like to analyze: [0] Bit shifts, [1] LANs and WANs, [2] Programming Languages.\n" \
                            "If you would like to summarize your own video, please enter 3: ")
    
    #Covert input to integer
    selection = int(selection)

    #Select the video from file explorer
    if selection == 3:

        #Retrieve input URL or empty line
        URL = input("Enter the URL of the Youtube video you would like to summarize. " 
                    "If you have the video downloaded and would like to choose from your file explorer instead, just hit enter: ")
        
        #Repeatedly ask for user input until a valid video is given
        while True:

            #If empty line, have user selected downlaoded video from file explorer
            if URL == "":
                print('Select the video you would like to summarize: ')
                Tk().withdraw()
                video = filedialog.askopenfilename()
                
                #Exit program if the user hits cancel when selecting a video from file explorer
                if video == "":
                    print("No video selected. Exiting Program.")
                    sys.exit()
                    
                break

            #Ensure link is from the offical Youtube domain
            elif r"https://www.youtube.com" in URL or r"https://youtu.be/" in URL:

                with YoutubeDL() as ydl:

                    #Extract the youtube video download path
                    info = ydl.extract_info(URL, download=False)

                    #Extract the title and change to mp3 format
                    title = Path(ydl.prepare_filename(info)).with_suffix('.mp3')
                    video = os.path.join(file_path, os.path.basename(title))

                    #Specify audio extraction and video format
                    ydl_opts = {'extract_audio': True, 
                                'format': 'bestaudio',
                                'outtmpl': video}  

                    URL = [URL]
                    
                    print("Downloading Youtube Video...")
                    with YoutubeDL(ydl_opts) as ydl2:
                        ydl2.download(URL)

                break

            URL = input("Enter the URL of the Youtube video you would like to summarize. " 
                "If you have the video downloaded and would like to choose from your file explorer instead, just hit enter: ")
        
    else:

        video = vids[selection]

    return video, selection


def configure_API_key_instr() -> None:
    """
    Output instructions on how to configure OpenAI API Key
    """
    print('\nNOTE: Add your OpenAI API Key to the system environment variables.')
    print('Generate a key here: https://platform.openai.com/api-keys')
    print('Directions to add API Key to System Environment: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety')
    print('A credit balance above $0 is REQUIRED. This model uses very little tokens (Average < $0.005 per run when summarizing one of the 3 preconfigured videos).')
    print('To view the price of GPT 5 Nano (the model being used), refer to this link: https://openai.com/api/pricing/\n')


def main() -> None:

    #Warning: API Key Required
    configure_API_key_instr()

    #Retrieve the video from input
    video, selection = retrieve_video()

    model = VideoSummarizer(video, selection)
    model.run()


if __name__ == "__main__":
    main()