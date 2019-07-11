import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN


def set_face(real, face, top, left):
    f = face.shape[0]
    c = face.shape[1]
    for i in range(f):
        for j in range(c):

            if face[i, j, 0] != 0 and face[i , j,1] != 0 and face[i, j, 2] != 0:
                real[top+i, left+j, 0] = face[i, j, 0]
                real[top + i, left + j, 1] = face[i, j, 1]
                real[top + i, left + j, 2] = face[i, j, 2]

    return real


top = 0
left = 0
factor = 0.80
cap = cv2.VideoCapture(0)
detector = MTCNN()

foto = cv2.imread('./cepe.png')

while True:

    ret, imagen = cap.read()

    dets = detector.detect_faces(imagen)

    cuadros = []
    for d in dets:

        left = int(d['box'][0] - d['box'][2] * factor)
        top = int(d['box'][1] - d['box'][3] * factor)
        heigh = int(d['box'][3] + d['box'][3] * factor)
        width = int(d['box'][2] + d['box'][2] * factor)

        if top < 0:
            top = 0
        if left < 0:
            left = 0

        cuadros.append([(left, top), (d['box'][0] + width, d['box'][1] + heigh), (0, 255, 0)])


    for c in cuadros:
        y = c[1][1] - c[0][1]
        x = c[1][0] - c[0][0]

        proof = cv2.resize(foto, dsize=(x, y))
        proof = proof[0:imagen.shape[0], 0:imagen.shape[1], :]
        cara = np.zeros((imagen.shape[0], imagen.shape[1], 3))

        if c[1][0] > cara.shape[1]:
            x = x - (c[1][0]-imagen.shape[1])
        if c[1][1] > cara.shape[0]:
            y = y - (c[1][1]-imagen.shape[0])
        cara[c[0][1]:c[1][1], c[0][0]:c[1][0], :] = proof[0:y, 0:x, :]
        cv2.imwrite("cara.png",cara)

        mask = cv2.inRange(cara, (0, 0, 0), (0, 0, 0))
        bk = cv2.bitwise_and(imagen, imagen, mask=mask)
        cara = cv2.imread('cara.png')
        imagen = cv2.bitwise_or(cara, bk)

    cv2.imshow('frame', imagen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
