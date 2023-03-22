import tensorflow_datasets as tfds
import settings as s
import numpy as np
from tensorflow_datasets.core import visualization
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam


class DataSet:

    def __init__(self):
        self.__ds, self.__ds_info = tfds.load(s.DATASET, split=['train[:10]', 'test'], with_info=True)
        self.__train = self.__ds[0]
        self.__test = self.__ds[1]

        train_numpy = np.vstack(tfds.as_numpy(self.__train))
        test_numpy = np.vstack(tfds.as_numpy(self.__test))

        self.__train_image = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
        self.__train_label = np.array(list(map(lambda x: x[0]['label'], train_numpy)))
        self.__test_image = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
        self.__test_label = np.array(list(map(lambda x: x[0]['label'], test_numpy)))

        # TODO: нормировка изображений
        self.__model = Sequential([
            layers.Dense(32, activation='relu', input_shape=self.__train_image[0].shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid', input_shape=self.__train_label.shape)
        ])

        self.__trained = True
        self.__epochs = s.MAX_EPOCHS
        self.__save_filename = s.SAVE_FILENAME

        self.__prediction = None
        self.__load_weights()

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------
    def __show_image(self, index):
        visualization.image_visualizer.ImageGridVisualizer().show(self.__test.skip(index).take(1),
                                                                  self.__ds_info,
                                                                  rows=1,
                                                                  cols=1)

    def __after_train_processing(self):
        print("[__after_train_processing] started")
        self.__prediction = self.__model.predict(self.__test_image)
        print("[__after_train_processing] save started")
        self.__model.save_weights(filepath=self.__save_filename)
        print("[__after_train_processing] save ended")
        self.__is_trained = True

    def __load_weights(self):
        print("[__load_weights] try to load weights")
        try:
            if self.__model.load_weights(filepath=self.__save_filename) is not None:
                print("[__load_weights] loaded successfully")
                self.__after_train_processing()
            print("[__load_weights] loaded failed")
        except:
            print("[__load_weights] loaded failed")
            return

    def __train_model(self):
        self.__model.compile(loss='binary_crossentropy',
                             optimizer=Adam(learning_rate=0.00024),
                             metrics=['binary_accuracy'])

        stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)

        self.__model.fit(self.__train_image, self.__train_label, batch_size=500, verbose=1,
                         epochs=self.__epochs, validation_split=0.2, callbacks=[stop])

        self.__after_train_processing()

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------

    def get_info(self):
        return self.__ds_info

    def get_data_train_numpy(self):
        return tfds.as_numpy(self.__train)

    def get_data_test_numpy(self):
        return tfds.as_numpy(self.__test)

    def get_data_train_frame(self):
        return tfds.as_dataframe(self.__train, self.__ds_info)

    def get_data_test_frame(self):
        return tfds.as_dataframe(self.__test, self.__ds_info)

    def get_data_train_list(self):
        return list(self.__train)

    def get_data_test_list(self):
        return list(self.__test)

    def show_benchmark(self):
        tfds.benchmark(self.__ds, batch_size=s.BATCH_SIZE)

    def get_test_length(self):
        return len(self.__test)

    def show_examples(self):
        tfds.show_examples(self.__train, self.__ds_info)

    def predict(self, index):
        print(f"[predict] predicted value is {index}")
        self.__show_image(index)

    def plot_accuracy(self):
        pass

    def plot_entropy_loss(self):
        pass

    def train(self):
        if self.__is_trained:
            return
        self.__train()
