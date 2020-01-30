#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules

# In[1]:


# show images inline
get_ipython().run_line_magic('matplotlib', 'inline')

# automatically reload modules when they have changed
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import glob

# use this to change which GPU to use
gpu = 1

# set the modified tf session as backend in keras
setup_gpu(gpu)


# ## Load RetinaNet model

# In[6]:


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('resnet50_csv_38.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'wound'}


# ## Run detection on example

# In[7]:


# load image
image = read_image_bgr('20120810_093836.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
        
    print(score)
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()


# In[9]:


for image in glob.iglob("/home/deploy/Documents/woundsee-ai/data/retinanet_wound_1/training/*.jpg"):
    #load image
    image_name = image
    image      = read_image_bgr(image)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
#     print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    is_show = True

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

#         if score >= 0.9:
#             is_show = True
        
        is_show = False
        
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    if is_show == True :
        print(image_name)
        print(score)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()


# In[ ]:


for image in glob.iglob("/home/deploy/Documents/woundsee-ai/data/retinanet_wound_1/training/*.jpg"):
    #load image
    image_name = image
    image      = read_image_bgr(image)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
#     print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    is_show = False

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        if score >= 0.88:
            print(score)
            is_show = True
        
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    if is_show == True :
        print(image_name)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()


# In[ ]:


def sort_list_boxs(list_boxs):
    for index in range(0, len(list_boxs)):
        list_box  = list_boxs[index]
        max_len   = size_box(list_box)
        max_index = index 
        
        for i in range(0, len(list_boxs)):
            if i != index:
                if max_len <= size_box(list_boxs[i]):
                    max_index = i
                    max_len   = size_box(list_boxs[i])

        temp                 = list_boxs[index]
        list_boxs[index]     = list_boxs[max_index]
        list_boxs[max_index] = temp
        
    return list_boxs
        
def size_box(box):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    
    return (x2 - x1) * (y2 - y1)
    
def is_not_box_duplicate(list_boxs, box, target_index, index_duplicate):
    
    xt1, yt1 = box[0], box[1]
    xt2, yt2 = box[2], box[3]
    
    x_center = (xt2 - xt1) / 2 + xt1
    y_center = (yt2 - yt1) / 2 + yt1
    
    is_not_box_duplicate = True
    
    for index in range(0, len(list_boxs)):
        if index != target_index and (index not in index_duplicate):
            list_box = list_boxs[index]

            x1, y1 = list_box[0], list_box[1]
            x2, y2 = list_box[2], list_box[3]

            #compare box
            result = None
            if size_box(box) <= size_box(list_box):
                if (x1 < xt1 and y1 < yt1) or (xt2 < x2 and yt2 < y2):
                    if (x1 < x_center and y1 < y_center) and (x_center < x2 and y_center < y2):
                        is_not_box_duplicate = False
                        break
                   
    return is_not_box_duplicate


# In[ ]:


for list_box in list_boxs:
    print(list_box)
    x1, y1 = list_box[0], list_box[1]
    x2, y2 = list_box[2], list_box[3]
    print(x1, y1)
    print(x2, y2)


# In[ ]:


for index in range(0, len(list_boxs)): 
   list_box = list_boxs[index]
   diff     = lambda l1,l2: [x for x in l1 if x not in l2]
   
   target_box           = create_box(list_box)
   is_not_box_duplicate = True
   
   print(list_box)
   break


# In[ ]:


for image in a:
    #load image
    image_name = image
    image      = read_image_bgr(image)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
#     print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    is_show = False

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        print(score)
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    print(image_name)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# In[ ]:


print(len(list_boxs[0]))


# In[ ]:




