import cv2
img = cv2.imread('/Users/dudu/PycharmProjects/tensorflow/data/image/u=1670324097,1966289191&fm=27&gp=0.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()