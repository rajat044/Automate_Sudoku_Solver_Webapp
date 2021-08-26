import cv2
import numpy as np
import matplotlib.pyplot as plt

#Taking input as grayscale image

def preprocessing(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #applying Gaussian blur to remove noise
    b_img = cv2.GaussianBlur(g_img, (9, 9), 0)
    #applying threshold algorithm for segmetation
    tre_img = cv2.adaptiveThreshold(b_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #inversing color
    inv_img = cv2.bitwise_not(tre_img, tre_img)
    #applying dilation
    #used to fill gaps and repairing gaps 
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    processed_img = cv2.dilate(inv_img, kernel)

    return processed_img

def find_corners(img):
    #cv2.RETR_EXTERNAL: gives outer contours only
    #cv2.contourArea : finds area of outermost polygon in img
    corners = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = corners[0] if len(corners) == 2 else corners[1]
    corners = sorted(corners, key = cv2.contourArea, reverse = True)
        
    for c in corners: 
        peri = cv2.arcLength(c, True)
        # cv2.approxPolyDP(curve, epsilon, closed[, approxCurve])
        # Curve-> hers is the largest contour
        # epsilon -> Parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation.
        # closed – If true, the approximated curve is closed. Otherwise, it is not closed.
        # approxPolyDP returns the approximate curve in the same type as the input curve
        apr = cv2.approxPolyDP(c, 0.015*peri, True)
        return apr if len(apr) == 4
def order_corners(corners):
    # Corners[0],... stores in format [[x y]]
    # Separate corners into individual points
    # Index 0 - top-right
    #       1 - top-left
    #       2 - bottom-left
    #       3 - bottom-right
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    tr, tl, bl, br = corners[0], corners[1], corners[2], corners[3]
    return tl, tr, br, bl

def per_transformation(img, corners):
#actuall order of corners = tr, tl, bl, br
#to move it clockwise order
    cor = order_corners(corners)
    tl, tr, br, bl = cor
    #calculating the width of sudoku
    #calculating the distance between br, bl and tr, tl 
    #taking maximum of them as width
    wiA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    wiB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1]))**2)

    width = max(int(wiA), int(wiB))

    #calculating the hight of sudoku
    #calculating the distance between tr, br and tl, bl 
    #taking maximum of them as width
    hiA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    hiB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1]))**2)

    height = max(int(hiA), int(hiB))

    dim = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype = 'float32')

    cor = np.array(cor, dtype = 'float32')

    grid = cv2.getPerspectiveTransform(cor, dim)

    return cv2.warpPerspective(img, grid, dsize = (width, height))
    #return cv2.resize(img, (width, height))
    
def create_image_grid(img):
    grid = np.copy(img)
    # not all sudoku out there have same width and height in the small squares so we need to consider differnt heights and width
    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9

    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding the cropped grid and inverting it
    grid = cv2.bitwise_not(grid, grid)


    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])

    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

    try:
        for i in range(9):
            for j in range(9):
                np.os.remove("BoardCells1/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("BoardCells1/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])

    return finalgrid


def scale_and_centre(img, size, margin=20, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))

def extract(img):
    processed_sudoku = processing(img)
    sudoku = find_corners(processed_sudoku)
    transformed = perspective_transform(img, sudoku)
    transformed = cv2.resize(transformed, (450, 450))
    sudoku = create_image_grid(transformed)
    return sudoku
