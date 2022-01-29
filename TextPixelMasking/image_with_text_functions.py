import numpy as np
import cv2 as cv
import string
import math

PRINTABLE_ARRAY = list(string.printable)
rng = np.random.default_rng(seed=1)
FONTS = (cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_PLAIN, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_COMPLEX, cv.FONT_HERSHEY_TRIPLEX, cv.FONT_HERSHEY_COMPLEX_SMALL, cv.FONT_ITALIC)

# just for debugging this file
def __log_(identifier, message):
    debug=False
    if (debug == True):
        module_name = "image_with_text_functions"
        print(module_name+"."+identifier, ":", message)


def put_random_text(img):
    font = FONTS[int(rng.integers(0, len(FONTS), 1))] #randomly select a font from the FONTS tuple
    topLeftCornerOfText = (int(rng.integers(0, img.shape[1], 1)), int(rng.integers(0, img.shape[0], 1))) #randomly select a place on the image
    fontScale = min(img.shape[0],img.shape[1])/(100/rng.random()) #randomly select a scale for the text
    fontColor = (255*(rng.random()**3),255*(rng.random()**3),255*(rng.random()**3)) #randomly select a color, weighted towards darker colors by cubing the random number [0,1]
    text = ''.join(rng.choice(PRINTABLE_ARRAY, size=rng.integers(1,50,1), shuffle=False)) #randomly create a string of printable characters, between 1 and 50 characters in length
    thickness = rng.integers(max(1, int(fontScale/2)),max(int(fontScale),2))

    cv.putText(img, text, topLeftCornerOfText, font, fontScale, fontColor, thickness, bottomLeftOrigin=False)

    (width, height), baseline =  cv.getTextSize(text, font, fontScale, thickness)

    return img, topLeftCornerOfText, width, height, baseline

    # print()
    # print("font", font)
    # print("text", text)
    # print("fontScale", fontScale)
    # print("thickness", thickness)

    # cv.imshow('image',img)
    # cv.waitKey(0)
    # print("\n\n\n")


def mask_image(img, topLeftCornerOfText, width, height, baseline, useGaussianNoise=False):
    mask = np.zeros(img.shape, np.uint8)
    # mask[topLeftCornerOfText[0]:topLeftCornerOfText[0]+box_size[1], topLeftCornerOfText[1]:topLeftCornerOfText[1]+box_size[0]]= (255,255,255)
    if not useGaussianNoise:
        mask[topLeftCornerOfText[1]-height-baseline:topLeftCornerOfText[1]+baseline, topLeftCornerOfText[0]:topLeftCornerOfText[0]+width]= (255,255,255)
        masked_image = mask | img

    else:
        meanPixelValues = np.mean(img, axis=(0,1))
        shape = (height+2*baseline, min(width, img.shape[1]-topLeftCornerOfText[0]), 3)
        noise = rng.normal(loc=meanPixelValues, size=shape)
        
        mask[topLeftCornerOfText[1]-height-baseline:topLeftCornerOfText[1]+baseline, topLeftCornerOfText[0]:topLeftCornerOfText[0]+width] = noise
        masked_image = np.copy(img)
        masked_image[topLeftCornerOfText[1]-height-baseline:topLeftCornerOfText[1]+baseline, topLeftCornerOfText[0]:topLeftCornerOfText[0]+width] = noise

    return masked_image, mask

def put_text_mask_image(img, useGaussianNoise=False):
    img, topLeftCornerOfText, width, height, baseline = put_random_text(img)
    masked_image, mask = mask_image(img, topLeftCornerOfText, width, height, baseline, useGaussianNoise)
    return masked_image, mask

def generate_crops_of_text_on_image_and_pixel_mask_from_path(path, x_size, y_size, n_channels):

    assert n_channels == 3, "Only n_channels == 3 supported"

    imgs = []
    masks = []
    raw_img = None
    __log_("path", path)
    if type(path)==type(""):
        raw_img = cv.imread(path, flags=cv.IMREAD_COLOR) #TODO change this for ALPHA channel support
        # __log_("raw_image", raw_img)
    else:
        print("path not string, but is", type(path))

    if(raw_img is not None):
        text_img, mask = put_text_and_mask_image_text_pixels_only(raw_img, x_size, y_size)

        # cv.imshow('img',text_img)
        # cv.imshow('mask',mask)

        num_x_crops = math.ceil(raw_img.shape[0]/x_size)
        num_y_crops = math.ceil(raw_img.shape[1]/y_size)
        __log_("num crops", num_x_crops*num_y_crops)

        for xi in range(num_x_crops):
            for yi in range(num_y_crops):
                mask_crop_temp = mask[xi*x_size : (xi+1)*x_size, yi*y_size : (yi+1)*y_size, :]
                appendThis = True
                if not np.any(mask_crop_temp): # then all pixels are zero
                    # print("np.any(mask_crop_temp)", np.any(mask_crop_temp))
                    appendThis = np.random.choice(a=[False, True], p=[0.9, 0.1]) # 10% chance to include blank crop
                    # print("appendThis", appendThis)
                if appendThis:
                    text_img_crop_temp = text_img[xi*x_size : (xi+1)*x_size, yi*y_size : (yi+1)*y_size, :]           
                    imgs.append(np.pad(text_img_crop_temp, ((0,x_size-text_img_crop_temp.shape[0]), (0,y_size-text_img_crop_temp.shape[1]), (0,0)), 'constant', constant_values=(0)))
                    masks.append(np.pad(mask_crop_temp, ((0,x_size-mask_crop_temp.shape[0]), (0,y_size-mask_crop_temp.shape[1]), (0,0)), 'constant', constant_values=(0)))
                    # cv.imshow('img_crop', imgs[-1])
                    # cv.imshow('mask_crop', masks[-1])
                    # cv.waitKey(0)

    return imgs, masks

def put_text_and_mask_image_text_pixels_only(img, local_rng=None):
    
    if local_rng is None:
        local_rng=rng

    #def put_random_text(img):
    font = FONTS[int(local_rng.integers(0, len(FONTS), 1))] #randomly select a font from the FONTS tuple
    topLeftCornerOfText = (int(local_rng.integers(0, img.shape[0]*0.5, 1)), int(local_rng.integers(20, img.shape[1], 1))) #randomly select a place on the image
    fontScale = 5 #(100/rng.integers(low=50, high=300)) #randomly select a scale for the text
    fontColor = (255*(1-local_rng.random()**10),255*(1-local_rng.random()**10),255*(1-local_rng.random()**10)) #randomly select a color, weighted towards darker colors by cubing the random number [0,1] //TODO change these values to be more realistic
    text = ''.join(local_rng.choice(PRINTABLE_ARRAY, size=local_rng.integers(1,50,1), shuffle=False)) #randomly create a string of printable characters, between 1 and 50 characters in length
    thickness = 5 #local_rng.integers(max(1, int(fontScale/2)),max(int(fontScale),2))

    cv.putText(img, text, topLeftCornerOfText, font, fontScale, fontColor, thickness, bottomLeftOrigin=False)

    mask = np.zeros(img.shape, np.uint8)
    cv.putText(mask, text, topLeftCornerOfText, font, fontScale+1, (255,255,255), thickness, bottomLeftOrigin=False) # put white text on black background to act as pixel mask

    return img, mask

#generate random crop of image then put text and mask
def generate_text_on_image_and_pixel_mask_from_path(path, x_size, y_size, n_channels, RNGseed=None):
    local_rng = None
    if RNGseed is not None:
        local_rng = np.random.default_rng(seed=RNGseed)
    else:
        local_rng=rng

    assert n_channels == 3, "Only n_channels == 3 supported"

    raw_img = None
    __log_("path", path)
    if type(path)==type(""):
        raw_img = cv.imread(path, flags=cv.IMREAD_COLOR) #TODO change this for ALPHA channel support
        # __log_("raw_image", raw_img)
    else:
        print("path not string, but is", type(path))

    if(raw_img is not None):
        # cv.imshow('img',text_img)
        # cv.imshow('mask',mask)

        num_x_crops = math.ceil(raw_img.shape[0]/x_size)
        num_y_crops = math.ceil(raw_img.shape[1]/y_size)
        __log_("num crops", num_x_crops*num_y_crops)
        xi = int(local_rng.integers(0, num_x_crops, size=1))
        yi = int(local_rng.integers(0, num_y_crops, size=1))

        raw_img_crop = raw_img[xi*x_size : (xi+1)*x_size, yi*y_size : (yi+1)*y_size, :]    
        raw_img_crop = np.pad(raw_img_crop, ((0,x_size-raw_img_crop.shape[0]), (0,y_size-raw_img_crop.shape[1]), (0,0)), 'constant', constant_values=(0))

        #TODO make sure that the text gets placed on the image properly
        text_img, mask = put_text_and_mask_image_text_pixels_only(raw_img_crop)

        # remap the mask from a color image to a 2-d vector with values 0,1.
        mask = mask[:,:,0]
        mask[mask==255] = 1

        return text_img, mask

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


if __name__ == "__main__":
    with open("C:/Users/maxan/Documents/MDSAI/CS686_AI/Project/code/data_flist/train_shuffled.flist") as training_list:
        img=None
        file_path=None
        doContinue = True
        while doContinue:
            try:
                file_path = training_list.readline().strip()
            except Exception as e:
                doContinue = False
                print("no lines in file", e)
            
            img = cv.imread(file_path)
            img, topLeftCornerOfText, width, height, baseline = put_random_text(img)
            masked_image, mask = mask_image(img, topLeftCornerOfText, width, height, baseline, useGaussianNoise=False)
            cv.imshow('image',img)
            cv.imshow('masked_image',masked_image)
            cv.imshow('mask',mask)

            cv.waitKey(0)

            print("test generate_crops_of_text_on_image_and_pixel_mask_from_path(file_path, 256, 256, 3)")
            imgs, masks = generate_crops_of_text_on_image_and_pixel_mask_from_path(file_path, 64, 64, 3)
            for i in range(len(imgs)):
                cv.imshow('imgs'+str(i),imgs[i])
            for i in range(len(masks)):
                cv.imshow('masks'+str(i),masks[i])
            cv.waitKey(0)


