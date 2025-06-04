import cv2
import ultralytics
from ultralytics import YOLO
# ultralytics.checks()
import numpy as np

model_container = YOLO('models/best_cont_det_yolo12m_batch16_20epoch.pt')
# model_container = YOLO('models/best_cont_det_yolo12m.pt')
# model_container = YOLO('models/best_cont_det_yolo9m_20epochs_dataset_06_05_2025.pt')
# model_container = YOLO('models/yolov9_container_segment.pt')
# model_container_number = YOLO('models/yolov8_container_number_detect.pt')
model_container_number = YOLO('models/cont_numb_yolov9t_1.pt')
# model_container_damage = YOLO('models/best_yolov9m_c4_practica.pt')
model_container_damage = YOLO('models/best_cont_dmg_yolo9m.pt')
# model_container_damage = YOLO('models/best.pt')


# классы для моделек, у чела из ультралитикс с ютуба было так
class ModelContainer():
    def __init__(self):
        self.device = 'cpu'
        self.model = self.load_model()

    def load_model(self):
        # model = YOLO('/content/yolov9_container_segment.pt')
        model = model_container
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame, conf=0.5)
        return results

    def plot_bbboxes(self, results, frame):
        # тут по результатм предсказания получаем координаты рамок и не только
        xyxys = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            current_xyxys = boxes.xyxy
            current_confidences = boxes.conf
            current_class_ids = boxes.cls

            xyxys.append(current_xyxys)
            confidences.append(current_confidences)
            class_ids.append(current_class_ids)

        return frame, xyxys, confidences, class_ids


class ModelContainerNumber():
    def __init__(self):
        self.device = 'cpu'
        self.model = self.load_model()

    def load_model(self):
        # model = YOLO('/content/yolov8_container_number_detect.pt')
        model = model_container_number
        model.fuse()
        return model

    def predict(self, frame, box):
        # тут получаем обрезанные изображения с контейнерами
        results_list = []
        imgs = self.crop_image(frame, box)

        # тут на обрезанных ищем номер и записываем результат
        # в список результатов
        for img in imgs:
            # results = self.model.predict(source=img, conf=0.25)
            results = self.model(img, conf=0.45)
            results_list.append(results)

        return results_list

    def crop_image(self, frame, box):
        # тут по полученным координатам рамок обрезаем исходное изображение
        # и заносим обрезанные в список обрезанных изображений, чтобы на них
        # искать номер
        imgs = []

        for xyxys in box:
            for xyxy in xyxys:
                img = np.array(frame)
                x_min, y_min, x_max, y_max = xyxy
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                cropped_img = img[y_min:y_max, x_min:x_max]
                imgs.append(cropped_img)

        return imgs

    def plot_bbboxes(self, results_list, frame):
        # тут по результатм предсказания получаем координаты рамок и не только
        xyxys = []
        confidences = []
        class_ids = []

        for results in results_list:
            for result in results:
                boxes = result.boxes.cpu().numpy()

                current_xyxys = boxes.xyxy
                current_confidences = boxes.conf
                current_class_ids = boxes.cls

                xyxys.append(current_xyxys)
                confidences.append(current_confidences)
                class_ids.append(current_class_ids)

        return frame, xyxys, confidences, class_ids


class ModelContainerDamage():
    def __init__(self):
        self.device = 'cpu'
        self.model = self.load_model()

    def load_model(self):
        # model = YOLO('/content/best_yolov9m_c4_practica.pt')
        model = model_container_damage
        model.fuse()
        return model

    def predict(self, frame, box):
        # тут получаем обрезанные изображения с контейнерами
        results_list = []
        imgs = self.crop_image(frame, box)

        # тут на обрезанных ищем номер и записываем результат
        # в список результатов
        for img in imgs:
            results = self.model.predict(source=img, conf=0.25)
            # results = self.model(img, conf=0.25)
            results_list.append(results)

        return results_list

    def crop_image(self, frame, box):
        # тут по полученным координатам рамок обрезаем исходное изображение
        # и заносим обрезанные в список обрезанных изображений, чтобы на них
        # искать номер
        imgs = []

        for xyxys in box:
            for xyxy in xyxys:
                img = np.array(frame)
                x_min, y_min, x_max, y_max = xyxy
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                cropped_img = img[y_min:y_max, x_min:x_max]
                imgs.append(cropped_img)

        return imgs

    def plot_bbboxes(self, results_list, frame):
        # тут по результатм предсказания получаем координаты рамок и не только
        xyxys = []
        confidences = []
        class_ids = []

        for results in results_list:
            for result in results:
                boxes = result.boxes.cpu().numpy()

                current_xyxys = boxes.xyxy
                current_confidences = boxes.conf
                current_class_ids = boxes.cls

                xyxys.append(current_xyxys)
                confidences.append(current_confidences)
                class_ids.append(current_class_ids)

        return frame, xyxys, confidences, class_ids
