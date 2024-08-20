from tensorflow.keras.models import load_model
import numpy as np
import pickle
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
pathModel='acgan_15all.h5'
generator = load_model(pathModel)

param = {"Max_A_Size": 15, "Max_B_Size": 15, "seed": 180,
         "dir": "GAN/dataset/dataset1/" }

dim = 250904
images = {}

f_myfile = open(param["dir"] + 'train_15x15_MI.pickle', 'rb')
images["Xtrain"] = pickle.load(f_myfile)
f_myfile.close()

f_myfile = open(param["dir"] + 'y_train.pickle', 'rb')
images["Classification"] = pickle.load(f_myfile)

(x_train, y_train) = np.asarray(images["Xtrain"]), np.asarray(images["Classification"])
print(x_train.shape)
print(y_train.shape)

noise_input = np.random.uniform(-1.0, 1.0, size=[dim, 200])

class_label = 0
noise_label = np.zeros((dim, 2))
noise_label[:, class_label] = 1
step = class_label
noise_input = [noise_input, noise_label]

generator.summary()
predictions = generator.predict(noise_input)
predictions = tf.reshape(predictions, [dim, 15, 15])

print(1, type(images["Xtrain"]))
print(2, type(predictions.numpy().tolist()))
new = predictions.numpy().tolist()
print(len(new))
images["Xtrain"].extend(new)
print(4, type(images["Classification"].tolist()))
print(4, type(list(np.zeros(dim))))

# Extend Classification with zeros
images["Classification"] = images["Classification"].tolist()
images["Classification"].extend([0] * dim)
print(len(images['Xtrain']))

f_myfile = open(param["dir"] + 'XTrain50A%.pickle', 'wb')
pickle.dump(images["Xtrain"], f_myfile)
f_myfile.close()

f_myfile = open(param["dir"] + 'YTrain50A%.pickle', 'wb')
pickle.dump(images["Classification"], f_myfile)
f_myfile.close()
