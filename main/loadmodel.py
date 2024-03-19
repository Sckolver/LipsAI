import os 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model(): 
    model = Sequential([
    Conv3D(128, 3, input_shape=(75,46,140,1), padding='same', activation='relu'), 
    MaxPool3D((1,2,2)), # Сжимаем 2x2
    Conv3D(256, 3, padding='same', activation='relu'),
    MaxPool3D((1,2,2)),
    Conv3D(75, 3, padding='same', activation='relu'),
    MaxPool3D((1,2,2)),
    TimeDistributed(Flatten()), # 75 для инпута в LSTM
    Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), # В обе стороны
    Dropout(.5), # Регуляризация
    Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
    Dropout(.5),
    Dense(41, kernel_initializer='he_normal', activation='softmax') # 75x41  
    ])

    model.load_weights('/Lip reader/checkpoint')

    return model

