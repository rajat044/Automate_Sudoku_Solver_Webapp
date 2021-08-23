import streamlit as st
import numpy as np
from img_processing import extract
from extract_numbers import extract_number_from_image
from sudoku_solver import solve
import cv2
import time

def display(grid):
    
    return '|---'*9 + '|\n' + '\n'.join(['|'+'|'.join([str(j) for j in i])+'|' for i in grid])

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Sudoku Solver')
uploaded_file = st.file_uploader("Choose an image", ["jpg","jpeg","png"])
st.write('Or')
use_default_image = st.checkbox('Use default Sudoku')

opencv_image = None
marked_image = None
ipmes = '|I|P|-|S|U|D|O|K|U|\n'
opmes = '|O|P|-|S|U|D|O|K|U|\n'


if use_default_image:
    opencv_image = cv2.imread('default.png')
    

elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

if opencv_image is not None:
    sudoku_image = extract(opencv_image)
    ex_digits = extract_number_from_image(sudoku_image)
    
    
    with st.spinner('Running a Neural Net to extract Sudoku from image'):
        time.sleep(2)
        st.success('Sudoku Has Been Successfully Extracted')

    #ex_digits = [[0 for _ in range(9)] for _ in range(9)]
    #print(ex_digits)
    st.markdown(ipmes + display(ex_digits))
    st.write('\n\n')


    with st.spinner('Sudoku is being solved'):
        time.sleep(1)
        sol = solve(ex_digits)

    if not sol: 
        st.error("**This Sudoku can't be solved**.")
    else: 
        st.title("**Solution**")
        st.markdown(opmes + display(ex_digits))
