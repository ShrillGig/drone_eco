from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import json
from zipfile import ZipFile
import shutil
import matplotlib.pyplot as plt


def unpack(orig_path, buffer, image_path, mask_path):

    index = 0
    height = width = 640
    train = Path("train")
    coco_file = train / "_annotations.coco.json"

    for zip_file in orig_path.glob('*.zip'): #извлекаем изображения из zip файла из Загрузки
        if zip_file.match("video_waste.v7*"):
            zip_path = orig_path / zip_file.name
            with ZipFile(zip_path, 'r') as zip_ref:
                for member in sorted(zip_ref.namelist()):
                    if member.startswith("train/"):
                        zip_ref.extract(member, buffer)

    for image in train.iterdir():
        if image.suffix == ".json":
            with open(coco_file, "r") as f:
                data = json.load(f)
                for index_mask, annotation in enumerate(data['annotations']):
                    for image_id in data['images']:
                        if annotation['image_id'] == image_id['id']: #в первом находятся координаты, а во втором имя файла
                            number = image_id['extra']['name'][5] #извлекаем индекс из имя файла для сохранения в определенной папке
                            mask_subfolder = mask_path / f"mask{number}"
                            mask_subfolder.mkdir(exist_ok=True)
                            #создаем полигоны из координат из словаря
                            polygon = np.array(data['annotations'][index_mask]['segmentation'][0], dtype=np.int32).reshape(-1, 2)
                            mask = np.zeros((height, width), dtype=np.uint8)
                            cv2.fillPoly(mask, [polygon], 255)
                            name = Path(image_id["extra"]["name"]).stem + ".png" #берем имя из image_id и сохраняем как избражение
                            mask_folder = mask_subfolder / name
                            cv2.imwrite(str(mask_folder), mask)
                            break
                print("Masks are ready")

        elif f"video{index}" in image.stem: #сверяем начинается ли имя файла с video
            image_folder = image_path / f"image{index}"
            shutil.copy(image, image_folder)
        else: #первое изображение с новым индексом не теряется, а сохраняется в новой папке
            index += 1
            image_folder = image_path / f"image{index}"
            if image_folder.is_dir(): #если папка создана
                pass
            else: # если папка не создана
                image_folder.mkdir(exist_ok=True)
                print(f"Directory:{image_folder} was successfully created")
            shutil.copy(image, image_folder)

    return


def iou_function(frame, ground):

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.cvtColor(ground, cv2.COLOR_BGR2GRAY)

    # значения пикселей с [0; 255] на [0;1]
    y_true = (gray_mask // 255).astype(np.uint8)
    y_pred = (gray_image // 255).astype(np.uint8)

    intersection = np.logical_and(y_pred, y_true).sum() #считаем только y_true = 1 и y_pred = 1
    union = np.logical_or(y_pred, y_true).sum() #считаем остальные
    iou = intersection / union

    return iou

def core(model, frame_image, frame_mask):

    iou_number = 0

    if frame_image.suffix.lower() == ".jpg":
        frame_image = cv2.imread(str(frame_image))
        frame_mask = cv2.imread(str(frame_mask))

        results = model.predict(frame_image, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0: #если есть предсказание в кадре

            x1, y1, x2, y2 = detections.xyxy[0].int().cpu().tolist()
            confidence = detections[0].conf.item()
            new_image = results[0].plot()
            name = model.names[int(detections[0].cls)] #имя, к которому обьект был отнесен

            gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            frame_roi = gray_image[y1:y2, x1:x2]  # динамическая рамка благодаря object detection для фокуса threshold
            blur = cv2.GaussianBlur(frame_roi, (5, 5), 0)
            adapt_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            black_image = np.zeros([640, 640], np.uint8)  # создаем пустое черное изображение 640х640
            black_image[y1:y2,x1:x2] = adapt_thresh  # накладываем на черное изображение рамку с threshold, чтобы фон не мешал при IoU
            color_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)

            iou_number = iou_function(color_image, frame_mask)
            cv2.putText(color_image, f"IoU:{iou_number:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),2)
            cv2.putText(color_image, f"Conf:{confidence:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),2)

        else: #если нет никакого предсказания в кадре
            new_image = frame_image
            black_image = np.zeros([640, 640], np.uint8)
            color_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
            confidence = 0
            name = None
            cv2.putText(color_image, f"IoU:{iou_number:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),2)
            cv2.putText(color_image, f"Conf:{confidence:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),2)

        return color_image, new_image, frame_mask, iou_number, confidence, name

def graphs(iou_stack, conf_stack, right_list, wrong_list, none_list, output):

    row = []

    for i, (iou, conf) in enumerate(zip(iou_stack, conf_stack)):
        buffer = [i + 1, iou, conf] #создаем массив из трех элементов
        row.append(buffer) #создаем массив из массивов

    plt.figure()
    columns = ["Object", "Mean IoU", "Frame Conf"]
    plt.axis("off")
    table = plt.table(cellText=row, colLabels=columns, cellLoc="center", loc="center")

    for (row, col), cell in table.get_celld().items(): #жирный шрифт
        if row == 0:
            cell.set_text_props(weight="bold")
    out_graph = output / "iou_conf.png"
    plt.savefig(out_graph)

    plt.figure()
    plt.axhline(y=210, color="red", linestyle="--", label="Maximum frames")
    plt.bar(range(1, len(right_list) + 1), right_list, color="green", label="Right prediction")
    plt.bar(range(1, len(wrong_list) + 1), wrong_list, bottom=right_list, color="red", label="Wrong prediction")
    plt.bar(range(1, len(none_list) + 1), none_list, bottom=np.array(right_list) + np.array(wrong_list), color="gray", label="No predicition")

    for i, v in enumerate(right_list):
        plt.text(i + 1, v / 2, str(v), ha='center', color="white")  #нижний ярус (зеленый)

    for i, v in enumerate(wrong_list):
        plt.text(i + 1, right_list[i] + v / 2, str(v), ha='center', color="black")  #центральный ярус (красный)

    for i, v in enumerate(none_list):
        plt.text(i + 1, right_list[i] + wrong_list[i] + v / 2, str(v), ha='center', color="black") #верхний ярус (серый)

    plt.ylim(0, 300)
    plt.legend()
    plt.title('YOLO frame-level detections (fine-tuned on custom data)')
    plt.xlabel('Videos')
    plt.ylabel('Frame count')
    out_graph = output / "frame_class.png"
    plt.savefig(out_graph)
    plt.close('all')


def main():

    min_conf = 0.557
    class_name = "waste"
    iou_mean_list = []
    conf_mean_list = []
    iou_list = []
    conf_list = []
    right_list = []
    wrong_list = []
    none_list = []

    orig_path = Path("C:/Users/F1/Downloads/")
    buffer = Path("C:/Users/F1/PycharmProjects/Simulator")
    folder_image = Path("images")
    folder_mask = Path("masks")
    output = Path("C:/Users/F1/Desktop/video/object detection")

    output.mkdir(exist_ok=True)
    folder_image.mkdir(exist_ok=True)
    folder_mask.mkdir(exist_ok=True)

    model_path = 'best.pt'
    model = YOLO(model_path, task="detect")

    unpack(orig_path, buffer, folder_image, folder_mask)

    #цикл по папкам
    for index, (images, masks) in enumerate(zip(folder_image.iterdir(), folder_mask.iterdir())):
        right = wrong = not_detected = 0
        output_image = output / f"video_image{index + 1}.mp4"
        output_mask = output / f"video_mask{index + 1}.mp4"
        output_thresh = output / f"video_thresh{index + 1}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_image = cv2.VideoWriter(str(output_image), fourcc, 30, (640, 640))
        out_mask = cv2.VideoWriter(str(output_mask), fourcc, 30, (640, 640))
        out_thresh = cv2.VideoWriter(str(output_thresh), fourcc, 30, (640, 640))

        #цикл по изображениям внутри одной папки
        for image, mask in zip(images.iterdir(), masks.iterdir()):
            color_image, new_image, frame_mask, iou, confidence, name = core(model, image, mask)
            #система условий о результатах предсказаний по имени и уверенности
            if name == class_name and confidence > min_conf:
                right += 1
            elif name is None and confidence == 0:
                not_detected += 1
            elif name != class_name or (confidence < min_conf and confidence != 0):
                wrong += 1

            iou_list.append(iou)
            conf_list.append(confidence)
            out_thresh.write(color_image)
            out_image.write(new_image)
            out_mask.write(frame_mask)

        out_thresh.release()
        out_image.release()
        out_mask.release()

        #средний iou и confidence в одном видео
        iou_mean_list.append(round(float(sum(iou_list) / len(iou_list)), 3))
        conf_mean_list.append(round(float(sum(conf_list) / len(conf_list)), 3))

        right_list.append(right)
        wrong_list.append(wrong)
        none_list.append(not_detected)
        print(f"Video{index + 1} is done")

    print(f"Right list:{right_list}")
    print(f"\nWrong list:{wrong_list}")
    print(f"\nNone_list:{none_list}")

    graphs(iou_mean_list, conf_mean_list, right_list, wrong_list, none_list, output)


if __name__ == "__main__":
    main()
