import numpy as np
import cv2 as cv
import string
import math

PRINTABLE_ARRAY = list(string.printable)
rng = np.random.default_rng(seed=1)
FONTS = (cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_PLAIN, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_COMPLEX, cv.FONT_HERSHEY_TRIPLEX, cv.FONT_HERSHEY_COMPLEX_SMALL, cv.FONT_ITALIC)


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

    raw_img = cv.imread(path, flags=cv.IMREAD_COLOR) #TODO change this for ALPHA channel support
    text_img, topLeftCornerOfText, width, height, baseline = put_random_text(raw_img)
    _ , mask = mask_image(img, topLeftCornerOfText, width, height, baseline)

    imgs = []
    masks = []
    print("raw_img.size:", raw_img.shape)
    num_x_crops = math.ceil(raw_img.shape[0]/x_size)
    num_y_crops = math.ceil(raw_img.shape[1]/y_size)

    for xi in num_x_crops:
        for yi in num_y_crops:
            img_crop = text_img[yi*y_size: yi*y_size + y_size, xi*x_size: xi*x_size + x_size]
            mask_crop = mask[yi*y_size: yi*y_size + y_size, xi*x_size: xi*x_size + x_size]
            imgs.append(img_crop)
            masks.append(mask_crop)

    return imgs, masks

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

