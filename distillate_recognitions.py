# coding: utf-8

import json
import os
import shutil
import time

import time
import multiprocessing 

TEXTS_PATH = '/media/storage/B.A./Recognition'
OUTPUT_PATH = "/media/storage/alekseychuk/LDA/recognition"
INPUT_FOLDERS=sorted(os.listdir(TEXTS_PATH))


def perform_distillation(folders):
    for i in folders:
        for file_name in os.listdir(TEXTS_PATH+f'/{i}'):
            try:
                if not f'{i}' in os.listdir(OUTPUT_PATH):
                    os.mkdir(OUTPUT_PATH + f'/{i}')
                with open(TEXTS_PATH + f'/{i}' + '/' + file_name.rsplit(".", 1)[0] + '.txt', 'r') as input_file:
                    words = input_file.read() 
                    words = words.replace('\nTranscription:', '').replace('диадох','диадок').replace('виадук','диадок')
                    pos = words.rfind('The processing time')
                    if pos >= 0:
                        words = words[:pos]
                    words = words.strip("\n")
                    if len(words) > 0:
                        with open(OUTPUT_PATH + f'/{i}' + '/' + file_name.rsplit(".", 1)[0] + '.txt', 'w') as output_file:
                            output_file.write(words)

            except Exception:
                pass
        print(f"finished transcribe folder {TEXTS_PATH + f'/{i}'}")


    
if __name__ == '__main__':
    processes = []

    p1 = multiprocessing.Process(target=perform_distillation, args=(range(1541,1558),))
    processes.append(p1)
    p1.start()
    print('Process 1 started.')
    
    p2 = multiprocessing.Process(target=perform_distillation, args=(range(1561,1581),))
    processes.append(p2)
    p2.start()
    print('Process 2 started.')
    
    p3 = multiprocessing.Process(target=perform_distillation, args=(range(1581,1606),))
    processes.append(p3)
    p3.start()
    print('Process 3 started.')
    
    for process in processes:
        process.join()
        
