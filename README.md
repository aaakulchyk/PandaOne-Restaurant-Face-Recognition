# PandaOne-Restaurant-Face-Recognition
Our demo face recognition system for restaurant business, created during open BSU + IBA + SAP hackathon.

Создаём новую папку и скачиваем проект:

```
$ mkdir PandaOne
$ cd PandaOne
$ mkdir panda_one
$ cd panda_one
$ git clone https://github.com/archaizzard/PandaOne-Restaurant-Face-Recognition.git
```

После этого создаём виртуальное окружение и устанавливаем зависимости:

```
$ python3 -m venv ../venv
$ source ../venv/bin/activate
$ pip install -r requirements.txt
```

Если нужно перетренировать модель (по умолчанию будет обучена распознавать все наши лица), закидываем в папку `face_detection/dataset/` и там создаём новую подпапку. Если будете перетренировывать модель, то учтите, что пустые папки она проигнорит, поэтому можно скачать в них картинки 50 Cent, если нужно зарегистрировать класс, но чтобы его не распознавало.

Для отображения в шаблоне надо закинуть ещё изображение под именем `00000.jpg` в `face_detection/static/face_detection/dataset/` и там в соответсвующую подпапку.

Чтобы перетренировать, находясь в папке `face_detection`, введите:

```
$ python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
$ python train_model.py --embeddings output/embeddings.pickle	--recognizer output/recognizer.pickle --le output/le.pickle
```
Когда всё готово, запускайте сервер:

```
$ python manage.py runserver localhost:8000
```

Если пишет, что порт занят, открываете `face_detection/templates/face_detection/face_recognition.html` и внизу (там где js) правите `const TEST_URL = 'http://localhost:8000/test/';`. Кстати, тестовый красный квадрат включается/выключается в `views.py` раскомментированием/комментированием 57 строки. Важно, чтобы комп был подключен к инету, иначе не сможет отправлять запросы с фронта на бэк.
