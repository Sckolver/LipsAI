from typing import List
import tensorflow as tf
import cv2
import os 

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# Используем tf.keras.layers.StringLookup чтобы превращать char в num и наоборот. 
# Например, char_to_num(['i', 'g', 'o', 'r']) вернет <tf.Tensor: shape=(4,), dtype=int64, numpy=array([ 9,  7, 15, 18])>
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="") # oov_token - значение для неизветсного токена
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
) # invert=True - num to char 

def load_video(path:str) -> List[float]: 
    # Записываем фреймы видео после грейскейла
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:]) # Изолирование положения губ 
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

# Загружаем слова и токенизируем
def load_words(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        everything = f.readlines() # Считываем строки 
    tokens = []
    for line in everything:
        line = line.split() # Разделяем временные значения и сами строки
        if line[2] != 'sil': # Убираем тишину
            tokens = [*tokens,' ',line[2]] # Добавляем в tokens через пробел
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# Загружаем и видео, и слова
def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = f'/Lip reader/data/s1/{file_name}.mpg'
    words_path = f'/Lip reader/data/alignments/s1/{file_name}.align'
    frames = load_video(video_path) 
    words = load_words(words_path)
    
    return frames, words
