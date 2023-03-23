import string

import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_datasets.core import visualization
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam


# TODO: spare MLP
# TODO: отдельный сейв на spare
class MLP:

    def __init__(self, dataset: string, epochs: int, learning_rate: float, batch_size: int, main_save_path: string,
                 with_info: bool = True, use_spare: bool = False, spare_save_path: string = None):
        self.__ds, self.__ds_info = tfds.load(dataset, split=['train', 'test'], with_info=True)
        self.__train = self.__ds[0]
        self.__test = self.__ds[1]

        # в numpy для разделения входа и выхода
        train_numpy = np.vstack(tfds.as_numpy(self.__train))
        test_numpy = np.vstack(tfds.as_numpy(self.__test))
        self.__train_image = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
        self.__train_label = np.array(list(map(lambda x: x[0]['label'], train_numpy)))
        self.__test_image = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
        self.__test_label = np.array(list(map(lambda x: x[0]['label'], test_numpy)))

        # корректная разбивка выхода для работы модели
        labels_number = len(np.unique(self.__train_label))
        self.__train_label = keras.utils.to_categorical(self.__train_label, labels_number)
        self.__test_label = keras.utils.to_categorical(self.__test_label, labels_number)

        # main model
        self.__main_save_filename = main_save_path
        self.__main_model = Sequential([
            layers.Rescaling(1. / 255, input_shape=self.__train_image[0].shape),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128),
            layers.Dense(labels_number, activation='sigmoid')
        ])

        # spare model
        if use_spare and not (spare_save_path is None):
            self.__spare_save_filename = spare_save_path
            self.__spare_model = Sequential([
                layers.Rescaling(1. / 255, input_shape=self.__train_image[0].shape),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128),
                layers.Dense(labels_number, activation='sigmoid')
            ])
        else:
            self.__spare_save_filename = None
            self.__spare_model = None

        self.__trained = False
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__with_info = with_info

        self.__main_prediction = None
        self.__load_weights()

        self.__print_models_summary()
        self.__show_benchmark()
        self.__show_examples()

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
        self.__main_prediction = self.__main_model.predict(self.__test_image)
        if not self.__trained:
            print("[__after_train_processing] main save started")
            self.__main_model.save_weights(filepath=self.__main_save_filename)
            print("[__after_train_processing] main save ended")
            if self.__spare_model is not None:
                print("[__after_train_processing] spare save started")
                self.__spare_model.save_weights(filepath=self.__spare_save_filename)
                print("[__after_train_processing] spare save ended")
        self.__is_trained = True
        print("[__after_train_processing] finished")

    def __load_weights(self):
        print("[__load_weights] try to load weights")
        try:
            if self.__main_model.load_weights(filepath=self.__main_save_filename) is not None:
                print("[__load_weights] loaded successfully")
                self.__trained = True
                self.__after_train_processing()
            print("[__load_weights] loaded failed")
        except Exception:
            print("[__load_weights] loaded failed")
            return

    def __train_main(self):
        self.__main_model.compile(loss='binary_crossentropy',
                                  optimizer=Adam(learning_rate=self.__learning_rate),
                                  metrics=['binary_accuracy'])

        history = self.__main_model.fit(self.__train_image,
                                        self.__train_label,
                                        batch_size=self.__batch_size,
                                        verbose=1,
                                        epochs=self.__epochs,
                                        validation_split=0.2,
                                        validation_data=(self.__test_image, self.__test_label))
        self.__plot_history(history.history['binary_accuracy'], history.history['val_binary_accuracy'],
                            history.history['loss'], history.history['val_loss'])

    def __train_spare(self):
        if self.__spare_model is None:
            return
        self.__spare_model.compile(loss='binary_crossentropy',
                                   optimizer=Adam(learning_rate=self.__learning_rate),
                                   metrics=['binary_accuracy'])

        history = self.__spare_model.fit(self.__train_image,
                                         self.__train_label,
                                         batch_size=self.__batch_size,
                                         verbose=1,
                                         epochs=self.__epochs,
                                         validation_split=0.2,
                                         validation_data=(self.__test_image, self.__test_label))
        self.__plot_history(history.history['binary_accuracy'], history.history['val_binary_accuracy'],
                            history.history['loss'], history.history['val_loss'])

    def __train_model(self):
        self.__train_main()
        self.__train_spare()
        self.__after_train_processing()

    @staticmethod
    def __get_predicted_class_index(arr):
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
        if self.__main_prediction is None:
            print("[__get_predicted_value] __prediction is None")
            return
        if index >= len(self.__main_prediction) or index < 0:
            print(f"[__get_predicted_value] index {index} is invalid")
        return self.__get_predicted_class_index(self.__main_prediction[index])

    def __plot_history(self, acc, val_acc, loss, val_loss):
        if not self.__with_info:
            return
        print("[__plot_history] get values")
        epochs_range = range(self.__epochs)

        print("[__plot_history] plot accuracy")
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        print("[__plot_history] plot loss")
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def __print_models_summary(self):
        if not self.__with_info:
            return
        print("[__print_models_summary] main model")
        self.__main_model.summary()
        if self.__spare_model is None:
            return
        print("[__print_models_summary] spare model")
        self.__spare_model.summary()

    def __show_benchmark(self):
        if not self.__with_info:
            return
        print(f"[__show_benchmark] dataset benchmark started with batch_size = {self.__batch_size}")
        tfds.benchmark(self.__ds, batch_size=self.__batch_size)

    def __show_examples(self):
        if not self.__with_info:
            return
        tfds.show_examples(self.__train, self.__ds_info)

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

    def get_test_length(self):
        return len(self.__test)

    def predict(self, index):
        predicted_value = self.__get_predicted_value(index)
        print(f"[predict] predicted value is {predicted_value}")
        self.__show_image(index)

    def train(self):
        if self.__trained:
            return
        self.__train_model()
