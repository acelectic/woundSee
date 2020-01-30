import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import shuffle

def get_list_dir(path, ignores=['.ipynb_checkpoints']):
    all_ = os.listdir(path)
    return [ p for p in all_ if p not in ignores ]

def get_log(file_path):
    def write(msg):
        with open( file_path, 'a' ) as f:
            f.write( '%s\n' % msg )
    
    with open(file_path, 'w'):
        pass
    
    return write

'''
Description:
change style of text

Parameter:
text is text to chagne style
style is new style of text
'''
def get_text(text, style):
    
    style = style.upper()
    
    _dict = {
       'PURPLE' : '\033[95m',
       'CYAN' : '\033[96m',
       'DARKCYAN' : '\033[36m',
       'BLUE' : '\033[94m',
       'GREEN' : '\033[92m',
       'YELLOW' : '\033[93m',
       'RED' : '\033[91m',
       'BOLD' : '\033[1m',
       'UNDERLINE' : '\033[4m',
    }
    
    if style not in _dict.keys():
        raise Exception( 'style must be %s.' % ', '.join(_dict.keys()) )
    
    return '%s%s\033[0m' % (_dict[style], text)


'''
Description:
Read image in term of numpy.ndarray

Parameter:
path is path_to_file
cmap is color model you want
'''
def read_image( path, cmap ):
    
    cmap = cmap.lower()
    
    def read_rgb( path ):
        bgr = cv2.imread( path, cv2.IMREAD_COLOR )
        rgb = cv2.cvtColor( bgr, cv2.COLOR_BGR2RGB )
        return rgb
    
    def read_grayscale( path ):
        grayscale = cv2.imread( path, cv2.IMREAD_GRAYSCALE )
        return grayscale
        
    _dict = {
        'gray' : read_grayscale,
        'rgb' : read_rgb
    }
    if cmap not in _dict.keys():
        raise Exception( 'cmap must be %s' % str(_dict.keys()) )
    func = _dict[cmap]
    img = func(path)
    return img


'''
Description:
Write image to device

Parameter:
path is path_to_file
image is image to write
'''
def write_image( path, image, cmap ):
    def write_rgb():
        bgr = cv2.cvtColor( image, cv2.COLOR_RGB2BGR )
        cv2.imwrite( path, bgr )
        
    def write_gray():
        cv2.imwrite( path, image )
        
    cmap = cmap.lower()
    _dict = {
        'gray' : write_gray,
        'rgb2bgr' : write_rgb
    }
    if cmap not in _dict.keys():
        raise Exception( 'cmap must be %s' % str(_dict.keys()) )
        
    _dict[cmap]()
    

'''
Description:
Show image by matplotlib.pyplot

Parameter:
img is numpy.ndarray
cmap is color model you want
'''
def show_image( img, cmap ):
    cmap = cmap.lower()
    _dict = {
        'gray' : plt.cm.gray,
        'rgb' : None,
    }
    
    if cmap not in _dict.keys():
        raise Exception( 'cmap must be %s' % str(_dict.keys()) )
    
    fig, ax = plt.subplots( 1 )
    ax.imshow( img, cmap=_dict[cmap] )
    plt.show()

'''
Description:
Check if path exists

Parameter:
path is specific path
'''
def is_exists( path ):
    return os.path.exists( path )


'''
Description:
Crop window from original image

Parameter:
image is ndarray
x1, y1 is coordinate of the left top corner
x2, y2 is coordinate of the right bottom corner
'''
def crop( image, x1, y1, x2, y2 ):
    height, width = image.shape[:2]
    if any( i < 0 for i in [x1, y1, x2, y2] ):
        raise Exception( 'Out of bound ( x1, y1, x2, y2 must >= 0 ) ')
    if not ( (x1 <= width and x2 <= width) and (y1 <= height and y2 <= height) ):
        raise Exception( 'Out of Bound (%d or %d) > %d or (%d or %d) > %d' % (x1, x2, width, y1, y2, height) )
    window = image[ y1:y2, x1:x2 ]
    return window


'''
Description:
Create sub-images

Parameter:
image is original image
window_shape is shape of window
stride is stride
'''
def sub_images(image, window_shape, stride):
    
    def get_coordinates( image_shape, window_shape, stride ):
        image_height, image_width = image_shape[:2]
        window_height, window_width = window_shape[:2]
        x1_list = sorted( set( range( 0, image_width-window_width, stride ) ) | { image_width-window_width } )
        y1_list = sorted( set( range( 0, image_height-window_height, stride ) ) | { image_height-window_height } )
        coordinates = [ ( (x, y), (x+window_width, y+window_height) ) for y in y1_list for x in x1_list ]
        return coordinates
    
    
    coordinates = get_coordinates( image.shape, window_shape, stride )
    areas = [ ( ((x1, y1), (x2, y2)), crop(image, x1, y1, x2, y2) ) for (x1, y1), (x2, y2) in coordinates  ]
    
    # Draw rectangle and show
#     image_2 = image.copy()
#     for p1, p2 in coordinates:
#         cv2.rectangle(image_2,p1,p2,(np.random.randint(low=0, high=256),np.random.randint(low=0, high=256),np.random.randint(low=0, high=256)),1)
#     show_image(image_2,'rgb')
    
    return areas


'''
Description:
Random crop areas in original image

Parameter:
image is original image
window_shape is shape of window
number_of_random is number of times to random
'''
def random_crop( image, window_shape, number_of_random, coor=False  ):
    
    def get_coordinates( image_shape, window_shape, n ):
        image_height, image_width = image_shape[:2]
        window_height, window_width = window_shape[:2]
        maximum = (image_width-window_width) * (image_height-window_height)
        if n > maximum:
            raise Exception("Cannot random unique area ( maximum of random is %d but number of random is %d. ) %s" % (maximum, n, image.shape) )
        x1_list = list(range(0, image_width-window_width))
        y1_list = list(range(0, image_height-window_height))
        coordinates = [ ( (x, y), (x+window_width, y+window_height) ) for y in y1_list for x in x1_list ]
        shuffle( coordinates )
        return coordinates[:n]
    
    coordinates = get_coordinates( image.shape, window_shape, number_of_random )
# Draw rectangle and show
#     image_2 = image.copy()
#     for p1, p2 in coordinates:
#         cv2.rectangle(image_2,p1,p2,(np.random.randint(low=0, high=256),np.random.randint(low=0, high=256),np.random.randint(low=0, high=256)),1)
#     show_image(image_2,'rgb')

    if coor:
        return [ ( ((x1, y1), (x2, y2)), crop(image, x1, y1, x2, y2) ) for (x1, y1), (x2, y2) in coordinates  ]
    
    return [ crop(image, x1, y1, x2, y2) for (x1, y1), (x2, y2) in coordinates  ]
    
    
'''
Description:
Reconstruct image from sub-images

Parameter:
areas is list of cooridate and cropped area
original_shape is shape of original image
'''
def reconstruct_image( areas, original_shape ):

    def merge_window_to_background( coordinate, window, background ):
        (x1, y1), (x2, y2) = coordinate
        old = background[y1:y2, x1:x2]
        temp = np.stack([ window, old ],axis=-1)
        new = np.nanmean( temp, axis=-1 )
        
        background[y1:y2, x1:x2] = new
        return background
    
    background = np.full(fill_value=np.NaN, shape=original_shape, dtype=np.float32 )
    for i, (coor, area) in enumerate(areas):
        background = merge_window_to_background( coor, area, background)
        
    return background.astype(np.float32)


'''
Description:
Resize image by keep ratio and fill unknown pixel by default value

Parameter:
image is image need to resize
size is size of output ( h, w )
fill value is value is filled into unknown pixel
'''
def resize_keep_and_fill( image, size, fill_value ):
    height, width = size
    resized_image, r = resize_image( image, height, width, True )
    h, w = resized_image.shape[:2]
    d = ( height, width ) if len(image.shape) < 3 else ( height, width, image.shape[2] )
    bg = np.full( d , fill_value, dtype=np.uint8 )
    bg[:h,:w] = resized_image
    return bg

def resize_image( image, height, width, keep_ratio=True  ):
    if keep_ratio:
        h, w = image.shape[:2]
        r = min( height/h, width/w )
        dim = ( int( w * r ), int( h * r) )
        resized_image = cv2.resize( image , dim )
        return resized_image, r
    return cv2.resize( image , (width, height) )
        

'''
Description:
Flip and rotate image

Parameter:
image is image need to process
'''
def flip_and_rotate(img):
    def rotate(img):
        return [ np.rot90( img, i ) for i in range(4)]

    rotate_1 = rotate(img)
    rotate_2 = rotate( np.fliplr(img) )

    return rotate_1 + rotate_2
