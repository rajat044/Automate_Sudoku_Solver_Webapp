# Automate-Sudoku-Solver-Webapp:snake:

## Descpription 

This is an automated [Streamlit](https://www.streamlit.io/) web app using python. It extracts the sudoku grid from the image, then obtains the digits from the sudoku grid, and with the help of a backtracking algorithm, solves the extracted sudoku. 
Here I am using [OpenCV](https://pypi.org/project/opencv-python/) to extract the sudoku grid from the image. Then a pre-trained [keras](https://keras.io/) model to extract the digits from the grid. 

**Note:** The keras model was trained on the [mnist dataset](https://en.wikipedia.org/wiki/MNIST_database). 

## Required Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required dependencies.

```bash
pip install cv2
pip install streamlit
pip install numpy 
pip install matplotlib
pip install keras 
```

## How to use 
1. Download the all the files 
2. Intall all dependencies 
3. Run the **main.py** python file 



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
