# import myspsolution as mysp
# mysp=__import__("my-voice-analysis")
import myprosody as mysp
from pydub import AudioSegment
import pandas as pd
import pickle
import re
import speech_recognition as sr
import re
import subprocess
import math
import datetime


def convert_mp3_to_wav():
    # files
    src = "Voice.mp3"
    dst = "Voice-Converted.wav"

    # convert wav to mp3
    sound = AudioSegment.from_file(src)
    sound.export(dst, format="wav")

    print("Conversion completed")

# convert_mp3_to_wav()

# Run myprosody
################################################################################################################
# Audio file to be analyzed needs to be put into the file directory "workingdirectoryX/myprosody/audioFiles"  #
################################################################################################################

p="Trial 1 - Audio Only"
c=r"/Users/qixianghe/PycharmProjects/Classroom-Teaching-Behaviours-ML/myprosody"

result_df = mysp.mysptotal(p, c)

total_pause_duration = float(result_df['original_duration'][0]) - float(result_df['speaking_duration'][0])
average_pause_duration = total_pause_duration / int(result_df['number_of_pauses'][0])
result_df['average_pause_duration'] = [average_pause_duration]

print(result_df)

# print("Speaking time without pauses:")
# speakingtime_wopause = mysp.myspst(p, c)
# print(mysp.myspst(p, c))
#
# print("Total speaking duration (including pauses and fillers)")
# totalspeakingtime =
# print(mysp.myspod(p, c))

result_df.to_csv(f"reportsourcefile_prosodyanalysisresults.csv", index=None)

audiofilename = p


# Get details of audio file
# Extract number of seconds
process = subprocess.Popen(['ffmpeg', '-i', f"{p}.wav"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout, stderr = process.communicate()
matches = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout.decode(),
                    re.DOTALL).groupdict()

print(matches['hours'])
print(matches['minutes'])
# print(matches['seconds'])
# print(float(matches['seconds']))
proc = float(matches['seconds'])
rounded_seconds = int(proc)
print(rounded_seconds)
# print(round(float(matches['seconds']),0))

total_seconds = (int(matches['hours']) * 60 * 60) + (int(matches['minutes']) * 60) + rounded_seconds
no_batches = math.ceil(total_seconds / 30)

print('Total seconds:', total_seconds)
print('No. of 30sec batches:', no_batches)

def transcribeaudio(audiofilename):
    # transcribe audio file
    # audiofilename = "Test Lecture with Questions"
    AUDIO_FILE = f"{audiofilename}.wav"
    print(AUDIO_FILE)

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        # r.adjust_for_ambient_noise(source)
        audio = r.record(source, offset=30, duration=30)  # read the entire audio file

        # print("Transcription: " + r.recognize_google(audio))

        transcribed_text = r.recognize_google(audio)
        print(f'Transcription of filename: {AUDIO_FILE}')
        print(transcribed_text)


    # Print the transcript
    file_name = f"{audiofilename}-transcription.txt"

    with open(file_name, "w") as file:
        # Write to the file
        file.write(transcribed_text)
        file.close()

def transcribe_split_audio(audiofilename):
    audiofilename = f"{audiofilename}"
    AUDIO_FILE = f"{audiofilename}.wav"  ## this means that any user needs to create a folder called splitaudiofiles...
    print("Audio file to be transcribed:", AUDIO_FILE)

    text_summary = []
    # use the audio file as the audio source
    # Run x for number of 30 sec batches in audio file

    # Calculate time taken for analyzing/showing frame

    for x in range(no_batches):
        start_time = datetime.datetime.now()
        # print(f'Transcription of filename: {AUDIO_FILE}')
        x+=1
        try:
            r = sr.Recognizer()
            print(f"Split {x}")
            with sr.AudioFile(AUDIO_FILE) as source:
                    r.adjust_for_ambient_noise(source)
                    audio = r.record(source,offset=x*30, duration=30)  # read the entire audio file

                    # print("Transcription: " + r.recognize_google(audio))

                    transcribed_text = r.recognize_whisper(audio, language="english")
                    # print(f'Transcription of filename: {AUDIO_FILE}')
                    print(transcribed_text)
                    print('No. questions:', transcribed_text.count("?"))

                    text_summary.append(transcribed_text)

            end_time = datetime.datetime.now()
            delta = end_time - start_time
            frames_left_to_analyze = no_batches - x
            print('Time taken to analyze audio batch:', delta.total_seconds())
            print('Estimated time left:',
                  f"{round((frames_left_to_analyze * delta.total_seconds() / 60), 1)} mins // {frames_left_to_analyze} frames left to analyze")



        except Exception as error:
            print('exit?')
            print(error)

    transcribed_text = ' '.join(text_summary)
    print(transcribed_text)

    count_questions = transcribed_text.count("?")

    file_name = f"{audiofilename}-transcription.txt"
    with open(file_name, "w") as file:
        # Write to the file
        file.write(transcribed_text)
        file.close()

    return count_questions


    # # Print the transcript
    # file_name = f"{audiofilename}-transcription.txt"
    #
    # with open(file_name, "w") as file:
    #     # Write to the file
    #     file.write(transcribed_text)

def analyze_transcription(audiofilename):
    # Create a list of phrases that may indicate question
    # Idea is to pair -> (when, where, why, how, what, which, who, whom, whose) PLUS (do, is, was, would, does, may, might, are, were)
    questionphrase_1 = ["when", "where", "why", "how", "what", "which", "who", "whom", "whose"]
    questionphrase_2 = ["do", "is", "was", "would", "does", "may", "might", "are", "were", "can", "could"]

    # Also add in some phrases that typically used to start questions in classroom <- more can be added, maybe machine learning?!
    misc_phrases = ["does anybody", "does it"]

    combined_phrases = []
    for word1 in questionphrase_1:
        for word2 in questionphrase_2:
            combined_phrases.append(f"{word1} {word2} ") # add in a space at the end so that it searches for the exact phrase (e.g. 'what do, instead of raising a count when it finds 'what does' <= part of phrase)

    combined_phrases.extend(misc_phrases)

    # Retrieve the transcription
    file_name = f"{audiofilename}-transcription.txt"  # automize through script afterwards
    with open(file_name) as file:
        lines = file.readline()
        print(lines)
        print(type(lines))

        # Check how many items in list, should only be one (e.g., one line of transcribed text)


    questions_count = 0
    for phrase in combined_phrases:
        print(phrase)
        # output = lines.find(phrase)

        occurences = re.findall(phrase, lines)
        print(occurences, len(occurences))

        # Add in the number of occurences into the running count of questions askd
        questions_count+=len(occurences)


    # Print number of possible questions asked in audio
    print("Questions asked by speaker:")
    print(questions_count)


def analyze_transcription_3rdparty(audiofilename):
    # Create a list of phrases that may indicate question
    # Idea is to pair -> (when, where, why, how, what, which, who, whom, whose) PLUS (do, is, was, would, does, may, might, are, were)
    questionphrase_1 = ["when", "where", "why", "how", "what", "which", "who", "whom", "whose"]
    questionphrase_2 = ["do", "is", "was", "would", "does", "may", "might", "are", "were", "can", "could"]

    # Also add in some phrases that typically used to start questions in classroom <- more can be added, maybe machine learning?!
    misc_phrases = ["does anybody", "does it"]

    combined_phrases = []
    for word1 in questionphrase_1:
        for word2 in questionphrase_2:
            combined_phrases.append(f"{word1} {word2} ") # add in a space at the end so that it searches for the exact phrase (e.g. 'what do, instead of raising a count when it finds 'what does' <= part of phrase)

    combined_phrases.extend(misc_phrases)

    # Retrieve the transcription
    file_name = f"{audiofilename} - Transcription by 3rd Party.txt"  # automize through script afterwards
    with open(file_name) as file:
        lines = file.readlines()
        print(lines)
        print(type(lines))

        newlines = []
        for x in lines:
            if x == "\n":
                continue
            y = x.strip("\n")
            newlines.append(y)

        textsummary = " ".join(newlines)
        print(textsummary)
        print(type(textsummary))

        # Check how many items in list, should only be one (e.g., one line of transcribed text)


    questions_count = 0
    for phrase in combined_phrases:
        print(phrase)
        # output = lines.find(phrase)

        occurences = re.findall(phrase, textsummary)
        print(occurences, len(occurences))

        # Add in the number of occurences into the running count of questions askd
        questions_count+=len(occurences)


    # Print number of possible questions asked in audio
    print("Questions asked by speaker:")
    print(questions_count)

    return questions_count


count_questions = transcribe_split_audio(audiofilename)
# transcribeaudio(audiofilename)
# analyze_transcription(audiofilename)

# questions_count = analyze_transcription_3rdparty(audiofilename)
result_df = pd.read_csv(f"reportsourcefile_prosodyanalysisresults.csv")

result_df['number_questions'] = [count_questions]

result_df.to_csv(f"reportsourcefile_prosodyanalysisresults.csv", index=None)



