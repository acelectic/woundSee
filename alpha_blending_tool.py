import numpy as np
import cv2

def countUnique(item):
    unique, unique_count = np.unique(item, return_counts=True)
    return {"unique":unique, "unique_count":unique_count, "shape" :item.shape}

def adv_blend_settings():
    correct_color = np.array([50, 250, 80], dtype=np.float32) # overlay color for correct 
                                                                                              # prediction
    miss_color = np.array([250, 50, 80], dtype=np.float32) # overlay color for missing prediction
    extra_color = np.array([80, 50, 250], dtype=np.float32) # overlay color for extraneous 
                                                                                           # prediction
    blend_factor = 0.25
    orig_factor = 1-blend_factor
    
    return correct_color, miss_color, extra_color, blend_factor, orig_factor

def adv_blend_masks(gt_img,pr_img):
    gt_mask = np.where(gt_img > 128, 1, 0)
    pr_mask = np.where(pr_img > 128, 1, 0)
    corr_mask = gt_mask & pr_mask
    miss_mask = np.where(pr_mask < gt_mask, 1, 0)
    extra_mask = np.where(pr_mask > gt_mask, 1, 0)
    
    return corr_mask, miss_mask, extra_mask


def blend_adv(raw_img,gt_img,pr_img):
  correct_color, miss_color, extra_color, blend_factor, orig_factor = adv_blend_settings()
  corr_mask, miss_mask, extra_mask = adv_blend_masks(gt_img,pr_img)

  alpha = blend_factor
  bg = 1.0 - alpha

  # Initial overlay is performed with the input image.
  overlay = np.where(corr_mask>0, 
                 bg*raw_img + alpha*(correct_color*corr_mask), raw_img)

  # Employ the previous overlay to blend with another mask.
  overlay = np.where(miss_mask>0, 
                   bg*overlay + alpha*(miss_color*miss_mask), overlay)

  # Final overlay is another blend over the previous visualization.
  overlay = np.where(extra_mask>0, 
                   (2*bg)*overlay + (1-2*bg)*(extra_color*extra_mask), overlay)+0.5
    
  return np.array(overlay,dtype=np.uint8)

# Common routine for each subplot (the purpose is to remove all ticks from axes).
def plot_img(img,plt):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)


# Basic ground-truth plotting method assuming we keep file paths in list gt_files
def plot_gt(i,plt):
    path = gt_files[i]
    img = plot(path,plt)
    return img

# Basic prediction plotting method assuming we keep file paths in list pr_files
def plot_prediction(i,plt):
    path = pr_files[i]
    img = plot(path,plt)
    return img

# Blend input with ground-truth and prediction with missing, extraneous colors.
# Like plot_gt and plot_prediction, we assume we keep paths for ground truths and 
#   predictions in lists gt_files and pr_files. Also, the paths for original 
#   inputs are stored in list input_paths.
def plot_blend_adv(i,plt): 
    raw_img = cv2.imread(input_paths[i], cv2.IMREAD_COLOR)
    gt_img = cv2.imread(gt_files[i], cv2.IMREAD_COLOR)
    pred_img = cv2.imread(pr_files[i], cv2.IMREAD_COLOR)
    overlay = blend_adv(raw_img,gt_img,pred_img)
    plot_img(overlay,plt)
