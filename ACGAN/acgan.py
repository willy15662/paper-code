import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import pickle
import time
import sys
sys.path.append("..")
import gan
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import experimental as mixed_precision

def train(models, data, params):
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, model_name = params

    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_labels = np.eye(y_train.shape[1])[np.arange(16) % y_train.shape[1]]
    train_size = x_train.shape[0]

    start_time = time.time()  # 開始計時

    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = y_train[rand_indexes]

        if len(fake_labels.shape) > 2:
            fake_labels = np.reshape(fake_labels, (batch_size, -1))

        fake_images = generator.predict([noise, fake_labels])

        if fake_images.shape[1:3] != real_images.shape[1:3]:
            fake_images = tf.image.resize(fake_images, (real_images.shape[1], real_images.shape[2]))

        x = np.concatenate((real_images, fake_images))
        labels = np.concatenate((y_train[rand_indexes], y_train[rand_indexes]))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0.0

        disc_loss_real = discriminator.train_on_batch(real_images, [y[:batch_size], y_train[rand_indexes]])
        disc_loss_fake = discriminator.train_on_batch(fake_images, [y[batch_size:], y_train[rand_indexes]])
        disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)
        log = "%d: [discriminator loss: %f, src_loss: %f, lbl_loss: %f, src_acc: %f, lbl_acc: %f]" % (
            i, disc_loss[0], disc_loss[1], disc_loss[2], disc_loss[3], disc_loss[4])

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        y = np.ones([batch_size, 1])
        adv_loss = adversarial.train_on_batch([noise, fake_labels], [y, fake_labels])
        log = "%s [adversarial loss: %f, src_loss: %f, lbl_loss: %f, src_acc: %f, lbl_acc: %f]" % (
            log, adv_loss[0], adv_loss[1], adv_loss[2], adv_loss[3], adv_loss[4])
        print(log)

        if (i + 1) % save_interval == 0:
            gan.plot_images(generator, noise_input=noise_input, noise_label=noise_labels, show=False, step=(i + 1), model_name=model_name)

    end_time = time.time()  # 結束計時
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    generator.save(model_name + ".h5")

def build_and_train_models():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    param = {"Max_A_Size": 15, "Max_B_Size": 15, "seed": 180,
             "dir": "GAN/dataset/dataset1/" }

    images = {}
    
    f_myfile = open(param["dir"] + 'train_' + str(param['Max_A_Size']) + 'x' + str(param['Max_B_Size']) + '_MI.pickle', 'rb')
    images["Xtrain"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'y_train.pickle', 'rb')
    images["Classification"] = pickle.load(f_myfile)
    f_myfile.close()
    x_train, y_train = np.asarray(images["Xtrain"]), np.asarray(images["Classification"])
    print(type(x_train))

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_labels)

    model_name = "acgan_15all"
    latent_size = 200
    batch_size = 128
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )
    
    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = gan.discriminator(inputs, num_labels=num_labels)
    optimizer = Adam(learning_rate=lr, decay=decay)
    loss = ['binary_crossentropy', 'categorical_crossentropy']
    discriminator.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    labels = Input(shape=label_shape, name='labels')
    generator = gan.generator(inputs, image_size, labels=labels)
    generator.summary()
    
    optimizer = Adam(learning_rate=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False
    adversarial = Model([inputs, labels],
                        discriminator(generator([inputs, labels])),
                        name=model_name)
    adversarial.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, model_name)
    
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    train(models, data, params)

def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 2
        noise_label = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_label = np.zeros((16, 2))
        noise_label[:,class_label] = 1
        step = class_label

    gan.plot_images(generator,
                    noise_input=noise_input,
                    noise_label=noise_label,
                    show=True,
                    step=step,
                    model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        generator.summary()
        class_label = None
        if args.digit is not None:
            class_label = args.digit
        test_generator(generator, class_label)
    else:
        build_and_train_models()
