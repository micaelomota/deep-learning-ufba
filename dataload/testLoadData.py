import loadDataset
import cv2

data, labels = loadDataset.loadData("data_part1/train/")


td, tl, vd, vl = loadDataset.splitValidation(data, labels, 10)

print(tl[0])

cv2.imshow('Imagem', td[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
