import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt

# Set true if want to test this file and check the results
t = False
'''
Hyperparameters

size: Dimension in which to resize the images
gamma: Value for Gamma Correction
eps: Constant used to avoid having a denominator = 0
cells_size: Cell size for Hog computation
block_size: Block size for Hog normalization
bl_size: Block shift for Hog normalization
'''
size = (64,64)
gamma = 2

eps = 1
cells_size = 8
cl_shift = cells_size

block_size = 2          
bl_shift = 1           

bin = np.array([0,20,40,60,80,100,120,140,160])

# Sobel Kernel for y-direction derivative
vkernel = np.array(([-1,0,1],
                    [-2,0,2],
                    [-1,0,1]), dtype="int")

# Sobel Kernel for x-direction derivative
hkernel = np.array(([-1,-2,-1], 
                    [0,0,0],
                    [1,2,1]), dtype="int")

# Kernel for sharpen the image
sharpen = np.array(([0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]),dtype = 'int')


'''---------------'''

def preprocess_image(img):

    frame = cv2.resize(img,size)
    if len(frame.shape) > 2:
        gf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gf = frame

    invGamma = 1 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8") 
    cf = cv2.LUT(gf, table)

    bf = cv2.GaussianBlur(cf,(3,3),0)
    sf = np.uint8(convolve(bf,sharpen))

    if t == True:
        cv2.imwrite("frames/2_grey.png",gf)
        cv2.imwrite("frames/3_norm.png",cf)
        cv2.imwrite("frames/4_blur.png",bf)
        cv2.imwrite("frames/5_sharp.png",sf)

    return sf

def pre_built_hog(frame):
    frame = preprocess_image(frame)
    fd, hog_image = skimage.feature.hog(frame, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True)  
    # cv2.imshow("frame",hog_image)
    return fd


def compute_hog(frame):

    frame = preprocess_image(frame)

    hgradient = convolve(frame,hkernel)
    vgradient = convolve(frame, vkernel)

    magnitude = np.sqrt(np.square(hgradient) + np.square(vgradient))

    orientation = np.arctan(vgradient/(hgradient+eps))
    orientation = np.rad2deg(orientation)
        
    xsize = frame.shape[1]
    ysize = frame.shape[0]
    
    x_cells_number = np.uint16(((xsize-cells_size)/cl_shift) + 1)
    y_cells_number = np.uint16(((ysize-cells_size)/cl_shift) + 1)
    
    cells = np.zeros([y_cells_number,x_cells_number,bin.size])

    # print ("Cells number: " + str(x_cells_number))

    for i in range (0,y_cells_number):
        for j in range(0,x_cells_number):
            hist = np.zeros(bin.size)
            for y in range(0,cells_size):
                for x in range(0,cells_size):
                    px_orientation = orientation[y+(cl_shift*i), x+(cl_shift*j)]
                    px_magnitude = magnitude[y+(cl_shift*i), x+(cl_shift*j)]

                    k = 0
                    ins = False
                    while k < (len(bin)-1) and ins == False:
                        if px_orientation<bin[k+1]:
                            hist[k] += ((bin[k+1]-px_orientation)/(bin[k]+eps)) * px_magnitude
                            hist[k+1] += ((px_orientation-bin[k])/(bin[k]+eps)) * px_magnitude
                            ins = True
                        k+=1
                    if ins == False:
                        hist[k] += px_magnitude
        
            cells[i,j] = hist

    x_block_sz = np.uint16(((cells.shape[1] - block_size)/bl_shift)+1)
    y_block_sz = np.uint16(((cells.shape[0] - block_size)/bl_shift)+1)
    
    hog = []
    for i in range(0,y_block_sz):
        for j in range(0,x_block_sz):
            block = []
            for y in range(0,block_size):
                for x in range(0,block_size):
                    block = block + cells[y+(bl_shift*i),x+(bl_shift*j)].tolist()
            
            block = block/np.sqrt(np.sum(np.power(block,2)) + 1)
            block = block.tolist()
            hog = hog + block   
    
    hog = np.array(hog)
    return hog
    
def convolve(frame, kernel):
    f_height,f_width = frame.shape
    k_height, k_width = kernel.shape
    pad = (k_width - 1)//2
    img = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((f_height, f_width), dtype="float32")
    for y in np.arange(pad, f_height + pad):
        for x in np.arange(pad, f_width + pad):
            roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
    output = skimage.exposure.rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("float32")
    return output

def print_hog(cells,fsx,fsy):
    hog_img = np.zeros([fsx,fsy], dtype=np.uint8)
    origin_h = cells_size//2
    mag_max = np.max(cells)
    for y in range(cells.shape[0]):
        for x in range(cells.shape[1]):
            magnitudes = cells[y,x]
            angle = 10
            angle_gap = 20
            for mag in magnitudes:
                angle_rad = np.deg2rad(angle)
                x1 = int(origin_h+(origin_h*x*2))
                y1 = int(origin_h+(origin_h*y*2))
                x2 = int(x1 + (np.cos(angle_rad)*(mag/mag_max))*5)
                y2 = int(y1 + (np.sin(angle_rad)*(mag/mag_max))*5)
                cv2.line(hog_img, (y1, x1), (y2, x2), int(255))
                angle += angle_gap
        
            None

    cv2.imshow("hog" ,hog_img)
    
def test():
    # frame = cv2.imread("frames/1_base.png")
    frame = cv2.imread("frames/1_base.png")
    pframe = preprocess_image(frame)

    hgradient = convolve(pframe,hkernel)
    vgradient = convolve(pframe,vkernel)

    grad = np.sqrt(np.square(hgradient) + np.square(vgradient))
    grad *= 255 / grad.max()
    cv2.imwrite("frames/h_gradient.png",hgradient)
    cv2.imwrite("frames/v_gradient.png",vgradient)
    plt.imshow(grad, cmap='gray')
    plt.show()

    # hog = compute_hog(frame)
    # phog = pre_built_hog(frame)
    # cv2.waitKey(0)

if t == True:
    test()