def create_dataset(wound_csv, dataset, feature_dir, label_dir, prediction_csv, augment_args=[], num=None, new=True ):
    
    initialize_dataset(dataset, feature_dir, label_dir, new)
    
    count       = 1
    image_desc  = get_dict_csv(prediction_csv)
    wound_paths = read_csv(wound_csv)
    
    bg_only     = []
    
    for wound_path in wound_paths:
        
        print(wound_path)
        x1          = wound_path[1]
        wound_path  = wound_path[0]
        image_name  = wound_path.split("/")[-1]
        _, _, files = next( os.walk(wound_path) )
        
        # get path images
        original_image_path     = os.path.join( wound_path, image_name + '.JPG' )
        granulation_image_paths = [ os.path.join( wound_path, p ) for p in files if 'Granulation' in p and not '.tif' in p ]
        slough_image_paths      = [ os.path.join( wound_path, p ) for p in files if 'Slough' in p and not '.tif' in p ]
        necrosis_image_paths    = [ os.path.join( wound_path, p ) for p in files if 'Necrosis' in p and not '.tif' in p ]
        epithelial_image_paths  = [ os.path.join( wound_path, p ) for p in files if 'Epithelial' in p and not '.tif' in p ]
        
        # read images
        original_image     = my_utils.read_image_rgb(original_image_path)
        granulation_images = [ my_utils.read_image_grayscale(p) for p in granulation_image_paths ]
        slough_images      = [ my_utils.read_image_grayscale(p) for p in slough_image_paths ]
        necrosis_images    = [ my_utils.read_image_grayscale(p) for p in necrosis_image_paths ]
        epithelial_images  = [ my_utils.read_image_grayscale(p) for p in epithelial_image_paths ]
        
        
        height, width, _   = original_image.shape
        original_image     = cutting_image(original_image, image_desc[image_name + x1][1:], dsize)
       
        # create bg for mask 
        granulation_image = np.full( fill_value=0, shape=(height, width), dtype=np.uint8 )
        slough_image      = np.full( fill_value=0, shape=(height, width), dtype=np.uint8 )
        necrosis_image    = np.full( fill_value=0, shape=(height, width), dtype=np.uint8 )
        epithelial_image  = np.full( fill_value=0, shape=(height, width), dtype=np.uint8 )
        
        for img in granulation_images:
            img = (img == 255) * 255
            granulation_image = (( granulation_image + img ) > 127 ) * 255

        for img in slough_images:
            img = (img == 255) * 255
            slough_image = (( slough_image + img ) > 127 ) * 255

        for img in necrosis_images:
            img = (img == 255) * 255
            necrosis_image = (( necrosis_image + img ) > 127 ) * 255

        for img in epithelial_images:
            img = (img == 255) * 255
            epithelial_image = (( epithelial_image + img ) > 127 ) * 255

        epithelial_image  = ( epithelial_image == 255 ) * 4
        slough_image      = ( slough_image == 255 ) * 3
        granulation_image = ( granulation_image == 255 ) * 2
        necrosis_image    = ( necrosis_image == 255 ) * 1

        bg   = np.full(fill_value = 0, shape = (height, width), dtype = np.uint8) * 0
        temp = np.stack([bg, necrosis_image, granulation_image, slough_image, epithelial_image], axis = -1)
        temp = np.argmax(temp, axis = -1)

        print(np.unique(temp))
        label_image = temp * color

        label_image = cutting_image_gray(label_image, image_desc[image_name + x1][1:], dsize)
        print(np.unique(label_image))
        
#         print(type(np.unique(label_image)))
        
        if np.unique(label_image) == np.ndarray([0]):
            bg_only.append(image_name + x1)
     
        def plot():
            f, axs = plt.subplots(1,5)

            axs[0].imshow( granulation_image, cmap='gray' )
            axs[1].imshow( slough_image, cmap='gray' )
            axs[2].imshow( necrosis_image, cmap='gray' )
            axs[3].imshow( unstate_image, cmap='gray' )
            axs[4].imshow( epithelial_image, cmap='gray' )

            f, axs = plt.subplots(1,6, figsize=(10,10))

            axs[0].imshow( granulation_image, vmin=0, vmax=255, cmap='gray' )
            axs[1].imshow( slough_image, vmin=0, vmax=255, cmap='gray' )
            axs[2].imshow( necrosis_image, vmin=0, vmax=255, cmap='gray' )
            axs[3].imshow( epithelial_image, vmin=0, vmax=255, cmap='gray' )
            axs[4].imshow( label_image, vmin=0, vmax=255, cmap='gray' )

            f = plt.figure(figsize=(10,10))
            plt.imshow( original_image )

            f = plt.figure(figsize=(10,10))
            
            plt.imshow( label_image, vmin=0, vmax=255, cmap='gray' )
            plt.show()
            
        original_images = None
        if num != None and count == num+1:
            plot()
            break
        else:
            name = ( "%d" % count ).zfill(5) + ".png"
            feature_image_path = os.path.join( feature_dir, name )
            label_image_path = os.path.join( label_dir, name )
            if new:
                my_utils.write_image_bgr( path=feature_image_path, bgr=original_image )
                my_utils.write_image_grayscale( path=label_image_path, image=label_image)
                
                if "colors" in augment_args :
                    original_images, label_images = color_image(original_image, label_image, feature_image_path, label_image_path)
                if "rotation" in augment_args :
                    if original_images == None:
                        rotation_image([original_image], [label_image], feature_image_path, label_image_path)
                    else :
                        rotation_image(original_images, label_images, feature_image_path, label_image_path)
                    
                
        f, axes       = plt.subplots(1, 2, figsize=(10, 4))

        f.suptitle("%d %s" % (count, image_name))

        axes[0].set_title("Feature")
        axes[0].imshow(original_image)

        axes[1].set_title("Label")
        axes[1].imshow(label_image, vmin=0, vmax=255, cmap='gray')
        plt.show()
        count += 1

    print(bg_only)