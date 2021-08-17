import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json, load_model

# Load the saved model
json_file = open('Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Model/model.h5")

print("Loaded saved model from disk.")
 
# evaluate loaded model on test data
def identify_number(image):
    image_resize = cv2.resize(image, (28,28))    # For plt.imshow
    image_resize_2 = image_resize.reshape(1,28,28,1)    # For input to model.predict_classes
#   cv2.imshow('number', image_test_1)
    loaded_model_pred = np.argmax(loaded_model.predict(image_resize_2 , verbose = 0), axis = -1)
#    print('Prediction of loaded_model: {}'.format(loaded_model_pred[0]))
    return loaded_model_pred[0]

def ExtractDigits(sudoku):
    sudoku = cv2.resize(sudoku, (450,450))

    # split sudoku
    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
            image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
            if image.sum() > 25000:
                grid[i][j] = identify_number(image)
            else:
                grid[i][j] = 0
    return grid.astype(int)
