import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from loadmodel import load_model

st.set_page_config(layout='wide')
with st.sidebar: # Сайдбар
    st.image('https://www.121captions.com/wp-content/uploads/2021/01/forensic-lip-reader.jpeg')
    st.title('Lip reader')
    st.info('This project is based on LipNet deep learning model!')


st.title('This lip reader can :blue[read lips] :sunglasses:')
all_videos = os.listdir('/Lip reader/data/s1')
video = st.selectbox('Choose video', all_videos)

c1, c2 = st.columns(2)

if all_videos:
    with c1:
        st.info(f'{video} is converted to mp4:', icon="🤖")
        file_path = os.path.join('..', 'data', 's1', video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        show_video = open('test_video.mp4', 'rb')
        video_bytes = show_video.read()
        st.video(video_bytes)
    
    
    with c2:
        st.info('Model input:')
        frames, words = load_data(tf.convert_to_tensor(f'data/s1/{video}'))
        frames_uint8 = tf.cast(frames*255, tf.uint8)
        frames_uint8 = tf.squeeze(frames_uint8, axis=-1)
        imageio.mimsave('animation.gif', frames_uint8.numpy(), fps=10)
        st.image('animation.gif', width = 1000)
        
        st.info('Model output:')
        model = load_model()
        pred = model.predict(tf.expand_dims(frames, axis = 0))
        decoded = tf.keras.backend.ctc_decode(pred, [75], greedy=True)[0][0].numpy()
        st.text(decoded)
        st.info('Decoded version:')
        st.text(tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8'))