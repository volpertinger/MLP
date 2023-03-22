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
        self.__ds, self.__ds_info = tfds.load(s.DATASET, split=['train[:1000]', 'test'], with_info=True)
        self.__train = self.__ds[0]
        self.__test = self.__ds[1]

        train_numpy = np.vstack(tfds.as_numpy(self.__train))
        test_numpy = np.vstack(tfds.as_numpy(self.__test))

        self.__train_image = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
        self.__train_label = np.array(list(map(lambda x: x[0]['label'], train_numpy)))

        self.__test_image = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
        self.__test_label = np.array(list(map(lambda x: x[0]['label'], test_numpy)))

        self.__train_label = keras.utils.to_categorical(self.__train_label, s.CLASSES_NUMBER)
        self.__test_label = keras.utils.to_categorical(self.__test_label, s.CLASSES_NUMBER)

        self.__model = Sequential([
            layers.Dense(32, activation='relu', input_shape=self.__train_image[0].shape),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid', input_shape=self.__train_label.shape)
        ])
        self.__model = Sequential([
            layers.Rescaling(1. / 255, input_shape=self.__train_image[0].shape),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128),
            layers.Dense(s.CLASSES_NUMBER, activation='sigmoid')
        ])

        self.__trained = False
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
        if not self.__trained:
            print("[__after_train_processing] save started")
            self.__model.save_weights(filepath=self.__save_filename)
            print("[__after_train_processing] save ended")
        self.__is_trained = True
        print("[__after_train_processing] finished")

    def __load_weights(self):
        print("[__load_weights] try to load weights")
        try:
            if self.__model.load_weights(filepath=self.__save_filename) is not None:
                print("[__load_weights] loaded successfully")
                self.__trained = True
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

        self.__model.fit(self.__train_image, self.__train_label, batch_size=s.BATCH_SIZE, verbose=1,
                         epochs=self.__epochs, validation_split=0.2, callbacks=[stop])

        self.__after_train_processing()

    def __get_predicted_class_index(self, arr):
        class_index = 0
        element_prediction = 0
        current_index = 0
        for el in arr:
            if el > element_prediction:
                element_prediction = el
                class_index = current_index
            current_index += 1
        return class_index

    def __get_predicted_value(self, index):
        if self.__prediction is None:
            print("[__get_predicted_value] __prediction is None")
            return
        if index >= len(self.__prediction) or index < 0:
            print(f"[__get_predicted_value] index {index} is invalid")
        return self.__get_predicted_class_index(self.__prediction[index])

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
        predicted_value = self.__get_predicted_value(index)
        print(f"[predict] predicted value is {predicted_value}")
        self.__show_image(index)

    def plot_accuracy(self):
        pass

    def plot_entropy_loss(self):
        pass

    def train(self):
        if self.__trained:
            return
        self.__train_model()
