import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D,  Dense, Flatten, Dropout
from tensorflow.keras.optimizers import  Adam
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

def CNN2(images, y, params=None):
    print(params)
    x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.2, stratify=y, random_state=100)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    image_size = x_train.shape[1]
    image_size2 = x_train.shape[2]

    x_train = np.reshape(x_train, [-1, image_size, image_size2, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size2, 1])

    kernel = params["kernel"] 
    inputs = Input(shape=(image_size, image_size2, 1))

    X = Conv2D(16, (kernel, kernel), activation='relu', name='conv0')(inputs)
    X = Dropout(rate=params['dropout1'])(X)
    X = Conv2D(32, (kernel, kernel), activation='relu', name='conv1')(X)
    X = Dropout(rate=params['dropout2'])(X)
    X = Conv2D(64, (kernel, kernel), activation='relu', name='conv2')(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(X)
    # X = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(X)
    X = Dense(2, activation='softmax', kernel_initializer='glorot_uniform')(X)

    model = Model(inputs, X)
    adam = Adam(params["learning_rate"])

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

    history = model.fit(x_train, y_train, epochs=params["epoch"], verbose=2, validation_data=(x_test, y_test), batch_size=params["batch"], callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=10), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)])
    model.load_weights('best_model.h5')

    y_test = np.argmax(y_test, axis=1)
    Y_predicted = model.predict(x_test, verbose=0, use_multiprocessing=True, workers=12)
    Y_predicted = np.argmax(Y_predicted, axis=1)

    cf = confusion_matrix(y_test, Y_predicted)
    f1_val = f1_score(y_test, Y_predicted)

    return model, {"balanced_accuracy_val": balanced_accuracy_score(y_test, Y_predicted) * 100, "F1_score_val": f1_val, "TP_val": cf[0][0], "FN_val": cf[0][1], "FP_val": cf[1][0], "TN_val": cf[1][1]}
