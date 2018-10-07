
def testDataloader():
    from src import dataloader
    import cv2

    data, labels, classes = dataloader.loadData("data_part1/train/")
    td, tl, vd, vl = dataloader.splitValidation(data, labels, 10)

    print(tl[0])

    cv2.imshow('Imagem', td[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testSaveModel():
    import numpy as np
    x = np.zeros((10, 10))
    np.save("w", x)


def testLoadModel():
    import numpy as np
    x = np.load("w.npy")
    print(x)

if (__name__ == '__main__'):
	#testDataloader()
    testSaveModel()
    testLoadModel()