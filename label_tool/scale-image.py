import glob, os
import cv2

new_width = 1980
src_dir = 'set'
dest_dir = 'scale-width-'+str(new_width)
os.makedirs(dest_dir, exist_ok=True)

for img_path in glob.glob(src_dir + '/*/*.jpg'):
    img = cv2.imread(img_path)

    img_dir_name = img_path.split(os.sep)[-2]
    os.makedirs(os.path.join(dest_dir, img_dir_name), exist_ok=True)

    scale = new_width / img.shape[1]

    img = cv2.resize(img, None, fx=scale, fy=scale)

    new_img_path = os.path.join(dest_dir, img_dir_name,
                                img_path.split(os.sep)[-1])
    cv2.imwrite(new_img_path, img)
