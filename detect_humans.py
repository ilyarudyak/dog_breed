import cv2
from data_prep import get_files


def is_human(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def get_accuracy(files, n=100):
    return sum([is_human(file) for file in files]) / n


if __name__ == '__main__':
    human_files, dog_files = get_files()
    human_files_short = human_files[:100]
    dog_files_short = dog_files[:100]
    print(get_accuracy(human_files_short))
    print(1 - get_accuracy(dog_files_short))