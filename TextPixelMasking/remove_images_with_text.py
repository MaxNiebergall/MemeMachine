import easyocr
import time
import numpy as np
import os

reader = easyocr.Reader(['en'])

image_file_paths = []
train_data_dir = 'D:/MemeMachine_ProjectData/dataset/training'
validation_data_dir = 'D:/MemeMachine_ProjectData/dataset/validation'

def image_has_text(file_path):
    result = reader.detect(file_path.strip())
    return any(result[0])

if __name__ == '__main__':
   
    for i, file_ in enumerate(os.listdir(train_data_dir+"/")):
        if i % 1000 == 999:
            print('i=',i)
        hasText = (image_has_text(train_data_dir+"/"+str(file_)))
        if hasText:
            os.remove(train_data_dir+"/"+str(file_))

    for file_ in  os.listdir(validation_data_dir+"/"):
        print(i, file_)
        hasText = (image_has_text(validation_data_dir+"/"+str(file_)))
        print(hasText)
        if hasText:
            os.remove(validation_data_dir+"/"+str(file_))