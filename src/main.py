from lib import *


def main():
    original_image = cv.imread('./res/stockimage3.jpg')

    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

    detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.2)
    # 1.3 originally

    for (column, row, width, height) in detected_faces:
        cv.rectangle(
            original_image,
            (column, row),
            (column + width, row + height),
            (255, 0, 0),
            4
        )

    small = cv.resize(original_image, (0, 0), fx=0.3, fy=0.3)

    cv.imshow('Image', original_image)  # original_image or small
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
