import pickle
import string
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from MLP import ModelSettings as ms
from tensorflow_datasets.core import visualization
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.neural_network import MLPClassifier


class MLP:

    def __init__(self, dataset: string, epochs: int, learning_rate: float, batch_size: int, main_save_path: string,
                 image_dimension: int, with_info: bool = True, use_spare: bool = False, spare_save_path: string = None):
        self.__ds, self.__ds_info = tfds.load(dataset, split=['train', 'test'], with_info=True)
        self.__train = self.__ds[0]
        self.__test = self.__ds[1]

        # в numpy для разделения входа и выхода
        train_numpy = np.vstack(tfds.as_numpy(self.__train))
        test_numpy = np.vstack(tfds.as_numpy(self.__test))
        self.__train_image_3d = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
        self.__train_label = np.array(list(map(lambda x: x[0]['label'], train_numpy)))
        self.__test_image_3d = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
        self.__test_label = np.array(list(map(lambda x: x[0]['label'], test_numpy)))
        # для spare метода нужен именно 2д массив
        self.__train_image_2d = self.__transform_3d_to_2d(self.__train_image_3d)
        self.__test_image_2d = self.__transform_3d_to_2d(self.__test_image_3d)

        # корректная разбивка выхода для работы модели
        labels_number = len(np.unique(self.__train_label))
        self.__train_label = keras.utils.to_categorical(self.__train_label, labels_number)
        self.__test_label = keras.utils.to_categorical(self.__test_label, labels_number)

        # main model
        self.__main_save_filename = main_save_path
        self.__main_model = Sequential([
            layers.Rescaling(1. / 255, input_shape=self.__train_image_3d[0].shape),
            layers.Conv2D(ms.L1, image_dimension, padding='same', activation=ms.COMMON_ACTIVATION),
            layers.MaxPooling2D(),
            layers.Conv2D(ms.L2, image_dimension, padding='same', activation=ms.COMMON_ACTIVATION),
            layers.MaxPooling2D(),
            layers.Conv2D(ms.L3, image_dimension, padding='same', activation=ms.COMMON_ACTIVATION),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(ms.L4),
            layers.Dense(labels_number, activation=ms.LAST_ACTIVATION)
        ])

        # spare model
        if use_spare and not (spare_save_path is None):
            self.__spare_save_filename = spare_save_path
            self.__spare_model = MLPClassifier(
                hidden_layer_sizes=(ms.L1, ms.L2, ms.L3, ms.L4),
                activation=ms.COMMON_ACTIVATION,
                solver=ms.SOLVER,
                learning_rate="constant",
                learning_rate_init=learning_rate,
                early_stopping=True,
                max_iter=epochs
            )
        else:
            self.__spare_save_filename = None
            self.__spare_model = None

        self.__trained = False
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__with_info = with_info

        self.__main_prediction = None
        self.__spare_prediction = None
        self.__load_weights()

        self.__print_models_summary()
        self.__show_benchmark()
        self.__show_examples()

    # ------------------------------------------------------------------------------------------------------------------
    # Static
    # ------------------------------------------------------------------------------------------------------------------
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

    @staticmethod
    def __transform_3d_to_2d(set_3d):
        x, y, z, d = set_3d.shape
        d2_train_dataset = set_3d.reshape((x, y * z * d))
        return d2_train_dataset

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
        self.__main_prediction = self.__main_model.predict(self.__test_image_3d)
        self.__spare_prediction = self.__spare_model.predict(self.__test_image_2d)
        if not self.__trained:
            print("[__after_train_processing] main save started")
            self.__main_model.save_weights(filepath=self.__main_save_filename)
            print("[__after_train_processing] main save ended")
            if self.__spare_model is not None:
                print("[__after_train_processing] spare save started")
                pickle.dump(self.__spare_model, open(self.__spare_save_filename, "wb"))
                print("[__after_train_processing] spare save ended")
        self.__is_trained = True
        print("[__after_train_processing] finished")

    def __load_weights(self):
        print("[__load_weights] try to load weights")
        try:
            if self.__main_model.load_weights(filepath=self.__main_save_filename) is not None:
                if self.__spare_model is not None:
                    self.__spare_model = pickle.load(open(self.__spare_save_filename, "rb"))
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

        history = self.__main_model.fit(self.__train_image_3d,
                                        self.__train_label,
                                        batch_size=self.__batch_size,
                                        verbose=1,
                                        epochs=self.__epochs,
                                        validation_split=0.2,
                                        validation_data=(self.__test_image_3d, self.__test_label))
        self.__plot_history_compare(history.history['binary_accuracy'], history.history['val_binary_accuracy'],
                                    history.history['loss'], history.history['val_loss'])

    def __train_spare(self):
        if self.__spare_model is None:
            return
        self.__spare_model.fit(self.__train_image_2d, self.__train_label)
        self.__plot_history(self.__spare_model.loss_curve_, self.__spare_model.validation_scores_)

    def __train_model(self):
        print("[__train_model] start main train")
        self.__train_main()
        print("[__train_model] start spare train")
        self.__train_spare()
        print("[__train_model] start after train")
        self.__after_train_processing()
        print("[__train_model] finished")

    def __get_predicted_value(self, prediction, index):
        if prediction is None:
            print("[__get_predicted_value] __prediction is None")
            return
        if index >= len(prediction) or index < 0:
            print(f"[__get_predicted_value] index {index} is invalid")
        return self.__get_predicted_class_index(prediction[index])

    def __plot_history(self, acc=None, loss=None):
        if not self.__with_info:
            return
        print("[__plot_history] get values")
        epochs_range = range(self.__epochs)

        print("[__plot_history] plot accuracy")

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc)
        plt.legend(loc='lower right')
        plt.title('Training Accuracy')

        print("[__plot_history] plot loss")
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss)
        plt.legend(loc='upper right')
        plt.title('Training Loss')

        plt.show()

    def __plot_history_compare(self, acc, val_acc, loss, val_loss):
        if not self.__with_info:
            return
        print("[__plot_history_compare] get values")
        epochs_range = range(self.__epochs)

        print("[__plot_history_compare] plot accuracy")

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        print("[__plot_history_compare] plot loss")
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
        print(self.__spare_model.get_params())

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

    def get_test_length(self):
        return len(self.__test)

    def predict(self, index):
        predicted_main = self.__get_predicted_value(self.__main_prediction, index)
        predicted_spare = self.__get_predicted_value(self.__spare_prediction, index)
        print(f"[predict] main predicted value is {predicted_main}; spare predicted value is {predicted_spare}")
        self.__show_image(index)

    def train(self):
        if self.__trained:
            return
        self.__train_model()
