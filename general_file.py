from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from zipfile import ZipFile
import shutil
import matplotlib.pyplot as plt


def unpack(orig_path, buffer, image_path):

    index = 0
    train = Path("train")

    for zip_file in orig_path.glob('*.zip'):
        if zip_file.match("video_waste.v8*"):
            zip_path = orig_path / zip_file.name
            with ZipFile(zip_path, 'r') as zip_ref:
                for member in sorted(zip_ref.namelist()):
                    if member.startswith("train/"):
                        zip_ref.extract(member, buffer)

    for image in train.iterdir():
        if f"video{index}" in image.stem:
            image_folder = image_path / f"image{index}"
            shutil.copy(image, image_folder)
        else:
            index += 1
            image_folder = image_path / f"image{index}"
            if image_folder.is_dir():
                pass
            else:
                image_folder.mkdir(exist_ok=True)
                print(f"Directory:{image_folder} was successfully created")
            shutil.copy(image, image_folder)

    return

def iou(frame, ground):
    # бинаризируем изображение
    frame = frame / 255
    ground = ground / 255

    # данные формулы применимы исключительно для сегментации
    tp = (frame * ground).sum()
    fp = ((frame + ground) - frame).sum()
    fn = ((frame + ground) - ground).sum()

    iou = tp / (tp + fp + fn)

    return iou


def core(model, frame_image, frame_mask):

    iou_number = 0

    if frame_image.suffix.lower() == ".jpg":
        frame_image = cv2.imread(str(frame_image))
        frame_mask = cv2.imread(str(frame_mask))

        results = model.predict(frame_image, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:

            x1, y1, x2, y2 = detections.xyxy[0].int().cpu().tolist()
            conf = detections[0].conf.item()
            new_image = results[0].plot()

            gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            frame_roi = gray_image[y1:y2, x1:x2]  # динамическая рамка благодаря object detection для фокуса threshold
            blur = cv2.GaussianBlur(frame_roi, (5, 5), 0)
            adapt_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            black_image = np.zeros([640, 640], np.uint8)  # создаем пустое черное изображение 640х640
            black_image[y1:y2,x1:x2] = adapt_thresh  # накладываем на черное изображение рамку с threshold, чтобы фон не мешал при IoU
            color_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)

            iou_number = iou(color_image, frame_mask)
            cv2.putText(color_image, f"IoU:{iou_number:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)
            cv2.putText(color_image, f"Conf:{conf:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)

        else:
            new_image = frame_image
            black_image = np.zeros([640, 640], np.uint8)
            color_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
            conf = 0
            cv2.putText(color_image, f"IoU:{iou_number:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),2)
            cv2.putText(color_image, f"Conf:{conf:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),2)

        return color_image, new_image, frame_mask, iou_number, conf


def input_data(images, masks, output, index, model):

    iou_list = []
    conf_list = []

    output_image = output / f"video_image{index}.mp4"
    output_mask = output / f"video_mask{index}.mp4"
    output_thresh = output / f"video_thresh{index}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_image = cv2.VideoWriter(str(output_image), fourcc, 30, (640, 640))
    out_mask = cv2.VideoWriter(str(output_mask), fourcc, 30, (640, 640))
    out_thresh = cv2.VideoWriter(str(output_thresh), fourcc, 30, (640, 640))

    for image, mask in zip(images.iterdir(), masks.iterdir()):
        color_image, new_image, frame_mask, iou, conf = core(model, image, mask)
        iou_list.append(iou)
        conf_list.append(conf)
        out_thresh.write(color_image)
        out_image.write(new_image)
        out_mask.write(frame_mask)

    out_thresh.release()
    out_image.release()
    out_mask.release()

    return iou_list, conf_list


def graphs(index, iou_stack, conf_stack):

    i = 0
    row = []

    while i < index:
        buffer = [i, iou_stack[i], conf_stack[i]]
        row.append(buffer)
        i += 1

    columns = ["Object", "Mean IoU", "Frame Count"]
    plt.axis("off")
    table = plt.table(cellText=row, colLabels=columns, cellLoc="center", loc="center")
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.show()


def main():

    index = 0
    iou_mean_list = []
    conf_mean_list = []

    orig_path = Path("C:/Users/F1/Downloads/")
    buffer = Path("C:/Users/F1/PycharmProjects/Simulator")
    folder_image = Path("images")
    folder_mask = Path("masks")
    output = Path("output_image")

    output.mkdir(exist_ok=True)
    folder_image.mkdir(exist_ok=True)
    folder_mask.mkdir(exist_ok=True)

    model_path = 'best.pt'
    model = YOLO(model_path, task="detect")

    unpack(orig_path, buffer, folder_image)

    for images, masks in zip(folder_image.iterdir(), folder_mask.iterdir()):
        iou_list, conf_list = input_data(images, masks, output, index, model)
        iou_mean_list.append(round(float(sum(iou_list) / len(iou_list)), 3))
        conf_mean_list.append(round(float(sum(conf_list) / len(conf_list)), 3))
        index += 1
        print(f"Video{index} is done")

    graphs(index, iou_mean_list, conf_mean_list)


if __name__ == "__main__":
    main()
