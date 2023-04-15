import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa



def preprocess_data(filepath):
    # audio_file = filepath
    y, sr = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=y,
                                   sr=sr,
                                   n_mels=128 * 2,)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(4, 1.75), frameon=False)
    # Plot the mel spectogram
    img = librosa.display.specshow(S_db_mel,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    ax.set_axis_off()
    file_name = "mel_data/"+str(filepath.rsplit('/')[-1]).split('.')[0]+".jpg"
    fig.savefig(file_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close(fig)
    print("data processed")
    return file_name

def model_predict(filepath):
    BATCH_SIZE = 32
    IMG_WIDTH, IMG_HEIGHT = 64, 64

    model = tf.keras.models.load_model('./cdd_model.h5')

    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=BATCH_SIZE)
    print(classes)
    
    return classes[0]