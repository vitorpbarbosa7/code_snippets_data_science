def readimg2tensor(path, w, h):
    import cv2
    def img2tensor(array,channels):
        tensor = array.reshape(1,array.shape[0], array.shape[1], channels)
        return tensor
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (w,h))
    image = img2tensor(array = image, channels = 1)
    image = image.astype('float32')
    image = image/255
    return image