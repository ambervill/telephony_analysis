# coding: utf-8

from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
import pymorphy2
import ffmpeg
import json
import os
import shutil
import time
from vosk import Model, KaldiRecognizer
import wave
import time
import multiprocessing 

PROJECT_PATH = os.getcwd()
AUDIO_RECORDINGS = PROJECT_PATH + '/critical_audio'
REPORT_PATH      = PROJECT_PATH + "/alekseychuk/LDA/critical_recognition"
NORMALIZED_REPORT_PATH      = PROJECT_PATH + "/alekseychuk/LDA/critical_normalized_recognition"
MODEL = Model("/media/storage/vosk-model-ru-0.10")
REPORT_LINE_WIDTH = 100
AUDIO_RECORDINGS_FOLDERS=sorted(os.listdir(AUDIO_RECORDINGS))


def perform_recognize(folders):
    for i in folders:
        for audio_name in os.listdir(AUDIO_RECORDINGS+f'/{i}'):
            print(f'Processing file {audio_name}')
            try:
                if os.path.exists(f'audio{i}.wav'):
                    os.remove(f'audio{i}.wav')
                if not os.path.exists(REPORT_PATH + f'/{i}'):
                    os.mkdir(REPORT_PATH + f'/{i}')
                if not os.path.exists(NORMALIZED_REPORT_PATH + f'/{i}'):
                    os.mkdir(NORMALIZED_REPORT_PATH + f'/{i}')                
                    
                if audio_name.endswith('.mp3') and not os.path.exists(REPORT_PATH + f'/{i}/' + audio_name.rsplit(".", 1)[0] + '.txt'):
                
                    audio_report = open(REPORT_PATH + f'/{i}' + '/' + audio_name.rsplit(".", 1)[0] + '.txt', 'w')

                    stream = ffmpeg.input(AUDIO_RECORDINGS+ f'/{i}/'+audio_name)
                    stream = ffmpeg.output(stream, f'audio{i}.wav')
                    stream = ffmpeg.overwrite_output(stream)
                    ffmpeg.run(stream)

                    wf = wave.open(f'audio{i}.wav', "rb")
                    rec = KaldiRecognizer(MODEL, wf.getframerate())

                    recognition_report_length = 0

                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            json_dict = json.loads(rec.Result())
                            transcript = json_dict['text']
                            for word in transcript.split():
                                if (len(word) + recognition_report_length > REPORT_LINE_WIDTH):
                                    audio_report.write('\n')
                                    recognition_report_length = 0
                                audio_report.write(word + ' ')
                                recognition_report_length += len(word) + 1
                        else:
                            rec.PartialResult()
                    audio_report.close()
                         
                    with open(REPORT_PATH + f'/{i}' + '/' + audio_name.rsplit(".", 1)[0] + '.txt', 'r') as file:
                        txt_in_file=''
                        for line in file:
                            if line[0]!='S' and line[0]!='T' and line[0]!='N':
                                txt_in_file=txt_in_file+line
                    words_in_text=word_tokenize(txt_in_file, language="russian")
                    stop_words=stopwords.words("russian")
                    filtered_words=[]
                    for token in words_in_text:
                        if token not in stop_words:
                            filtered_words.append(token)
                    morph = pymorphy2.MorphAnalyzer()
                    for word in range(len(filtered_words)):
                        filtered_words[word] = morph.parse(filtered_words[word])[0].normal_form
                    end_txt=''
                    for word in filtered_words:
                        end_txt=end_txt+' '+word

                    with open(NORMALIZED_REPORT_PATH + f'/{i}' + '/' + audio_name.rsplit(".", 1)[0] + '.txt', 'w') as file:
                              file.write(end_txt)

                    os.remove(f'audio{i}.wav')
                    print(f'Finished file {audio_name}')
                    print("------------------------------------")
            except Exception:
                pass
        print(f"finished transcribe folder {AUDIO_RECORDINGS+f'/{i}'}")


    
if __name__ == '__main__':
    processes = []

    p1 = multiprocessing.Process(target=perform_recognize, args=([2000],))
    processes.append(p1)
    p1.start()
    print('Process 1 started.')
    
    p2 = multiprocessing.Process(target=perform_recognize, args=([1566,1567,1568,1569],))
    processes.append(p2)
    p2.start()
    print('Process 2 started.')
    
    for process in processes:
        process.join()
        
