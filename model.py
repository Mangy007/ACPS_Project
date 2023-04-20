import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
# import sox
import os
import soundfile as sf  # Missing import
from pydub import AudioSegment
import shutil

def split_audio(filepath):
    curr_dir = os.getcwd()
    out_dir = r'/split_audio'
    # assign files
    try:
        os.remove("split_audio")
        os.remove(os.path.basename(filepath[:-4]) + ".wav")
    except:
        print("No wav file exists.")
    os.chdir(os.path.join(curr_dir, 'audio_data'))
    input_file = os.path.basename(filepath)
    output_file = os.path.basename(filepath[:-4]) + ".wav"
    # convert mp3 file to wav file
    sound = AudioSegment.from_mp3(input_file)
    sound.export(output_file, format="wav")
    os.remove(input_file)
    os.chdir(curr_dir)

    filepath = filepath[:-4] + ".wav"

    y, sr = librosa.load(filepath)
    # y = librosa.effects.split(y_noisy, top_db=20) #remove silent (<20 dB) parts
    os.makedirs("."+out_dir, exist_ok=True)
    segment_dur_secs = 1 
    segment_length = sr * segment_dur_secs
    
    num_sections = int(np.ceil(len(y) / segment_length))

    split = []

    for i in range(num_sections):
        t = y[i * segment_length: (i + 1) * segment_length]
        split.append(t.astype("float64"))

    split_audio_dir = os.path.join(curr_dir, out_dir)
    os.chdir("."+split_audio_dir)
    for i in range(num_sections):
        recording_name = os.path.basename(filepath[:-4])
        out_file = f"{recording_name}_{str(i)}.wav"
        sf.write(out_file, split[i], sr)
        print("Write successful.")
    os.chdir(curr_dir)
    print("Data split.")

def preprocess_data(filepath) -> list:
    curr_dir = os.getcwd()
    split_audio(filepath)
    out_dir = r'/split_audio'
    file_name = []
    for file in os.listdir("."+out_dir):
        fp=""
        if file.endswith(".wav"):
            fp = os.path.join("."+out_dir, file)

        # audio_file = filepath
        y, sr = librosa.load(fp)
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
        f_name = "mel_data/"+str(fp.rsplit('/')[-1]).split('.')[0]+".jpg"
        fig.savefig(f_name, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)
        file_name.append(f_name)
    print("Data processed.")
    try:
        shutil.rmtree(os.path.join(curr_dir, 'split_audio'), ignore_errors=True)
        print("Deleted split_audio.")
    except:
        print("Could dnot delete split_audio.")
    
    return file_name

def model_predict(filepath_list):
    BATCH_SIZE = 32
    IMG_WIDTH, IMG_HEIGHT = 64, 64

    sum = 0

    model = tf.keras.models.load_model('./cdd_model.h5')
    for filepath in filepath_list:
        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMG_WIDTH, IMG_HEIGHT))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=BATCH_SIZE)
        # print(classes) #[x, y] where x is the probability of class="Distress", and y is the probability of class="Not Distress"
        if(classes[0][0]>0.5):
            sum+=1
    
    print("Sum = ",  sum)

    return sum