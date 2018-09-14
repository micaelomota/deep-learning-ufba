import os
import numpy as np
import cv2
from helpers import dataloader

#color = cv2.imread('nome.jpg', cv2.IMREAD_COLOR)
#gray = cv2.imread('nome.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('data_part1/test/14734.png', cv2.IMREAD_UNCHANGED)
# visualiza imagem até que uma tecla seja pressionada
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# salva imagem
#cv2.imwrite('saida.png', img)

data, labels = dataloader.loadData("data_part1/train/")
td, tl, vd, vl = dataloader.splitValidation(data, labels, 10)

print(tl[0])

cv2.imshow('Imagem', td[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
