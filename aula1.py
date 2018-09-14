import os
import numpy as np
import cv2

# carrega uma imagem colorida (3 canais, BGR)
#color = cv2.imread('nome.jpg', cv2.IMREAD_COLOR)

# carrega uma imagem em tons de cinza (1 canal)
#gray = cv2.imread('nome.jpg', cv2.IMREAD_GRAYSCALE)

# carrega uma imagem como ela é (X canais)
#img = cv2.imread('data_part1/test/14734.png', cv2.IMREAD_UNCHANGED)

# visualiza imagem até que uma tecla seja pressionada
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# salva imagem
#cv2.imwrite('saida.png', img)

test_path = "data_part1/train/"

folders = os.listdir(test_path)

height = 77
width = 71
num_channels = cv2.IMREAD_COLOR
num_images = 12

tensor4d = np.empty([num_images, height, width, num_channels], dtype=np.uint8)

for folder in folders: 
    print ("carregando pasta " + folder)
    
    files = os.listdir(test_path + folder)
    print(str(len(files)) + " arquivos encontrados")



quit()



for img in lista:
    tensor4d.append(cv2.imread(test_path + img, cv2.IMREAD_COLOR))
