import numpy as np
import cv2
import os

def check_path(path, display=False):
    exists = os.path.exists( path )
    if not exists:
        raise Exception( "%s not found!" % path )
    if display:
        print( "%s is OK." % path )
        
        
def read_image_rgb( path ):
    bgr = cv2.imread( path , cv2.IMREAD_COLOR )
    rgb = cv2.cvtColor( bgr , cv2.COLOR_BGR2RGB )
    return rgb


def read_image_grayscale( path ):
    grayscale = cv2.imread( path , cv2.IMREAD_GRAYSCALE )
    return grayscale


def write_image_bgr( path, bgr ):
    rgb = cv2.cvtColor( bgr, cv2.COLOR_RGB2BGR )
    return cv2.imwrite( path, rgb )


def write_image_grayscale( path, image ):
    return cv2.imwrite( path, image )


def get_window( row, col, image, window_width, window_height ):
    half_window_height = window_height // 2
    half_window_width = window_width // 2
    
    start_row = row - half_window_height
    start_col = col - half_window_width
    
    end_row = row + half_window_height
    end_col = col + half_window_width
    
    return image[ start_row:end_row, start_col:end_col]


def get_pairs( image_shape, window_height, window_width, stride):
    height, width = image_shape[:2]
    
    row_list = list( range( window_height // 2, height - window_height // 2 + 1, stride ) )
    col_list = list( range( window_width// 2, width - window_width // 2 + 1, stride ) )
    
    if height - max(row_list) - window_height // 2 > 0 :
        row_list.append( height - window_height // 2 )
        
    if width - max(col_list) - window_width // 2 > 0:
        col_list.append( width - window_width // 2 )
        
    pairs = [ (r,c) for r in row_list for c in col_list ]
    return pairs


def get_pairs_and_windows( image, window_height, window_width, stride ):
    pairs = get_pairs( image.shape, window_height, window_width, stride )
    windows = [ get_window( r, c, image, window_width, window_height ) for r, c in pairs ]
    return pairs, windows
