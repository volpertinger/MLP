# Multi Layer Perceptron

Реализован через Keras (основной способ) и SciKit (дополнительный).
Реализована именно классификация изображений.

## Dataset

Изображения людей и лошадей.

## Result

Обученная модель и графики accuracy, loss.

## Функционирование

При запуске main модель обучается или загружает сохраненные веса.
Затем при вводе индексов валидационной выборки можно протестировать
качество обучения модели.

## UserSettings

Предназначены для настройки пользователем.
При изменении параметров в настройках можно изменить датасет,
гиперпараметры обучения, режим обучения (через одну модель или с дополнительной)
и другое.

## ModelSettings

Предназначены для тонкой настройки моделей MLP. 
Лучше не трогать.