import cv2
import numpy as np
from keras.models import load_model
from img_processing import scale_and_centre


def predict(img_grid):
    image = img_grid.copy()
    
    image = cv2.resize(image, (28, 28))
    
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255
    
    model = load_model('Model/model.h5')
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)

    return pred.argmax()

def extract_number_from_image(img_grid):
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):

            image = img_grid[i][j]
            image = cv2.resize(image, (28, 28))
            original = image.copy()

            thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
            # threshold the image
            gray = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

            # Find contours
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                if (x < 3 or y < 3 or h < 3 or w < 3):
                    # Note the number is always placed in the center
                    # Since image is 28x28
                    # the number will be in the center thus x >3 and y>3
                    # Additionally any of the external lines of the sudoku will not be thicker than 3
                    continue
                ROI = gray[y:y + h, x:x + w]
                ROI = scale_and_centre(ROI, 120)
                
                tmp_sudoku[i][j] = predict(ROI)

    return tmp_sudoku

