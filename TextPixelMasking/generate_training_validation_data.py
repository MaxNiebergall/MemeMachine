#see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

from cProfile import label
import os
from torch import float32
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from image_with_text_functions import generate_text_on_image_and_pixel_mask_from_path
import numpy as np
import cv2 as cv
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import string 
import math

class CustomImageDataset(Dataset):
    #Transform options: 'random_crop', TBA 'detect-resize'
    def __init__(self, img_dir, x_size, y_size, n_channels=3, transform=None, RNGseed=None, target_transform=None):
        self.x_size = x_size
        self.y_size = y_size
        self.n_channels = n_channels
        self.img_dir = img_dir
        self.transform = transform 
        self.target_transform = target_transform
        self.img_paths = []
        self.RNGseed = RNGseed
        for file_ in  os.listdir(img_dir+"/"):
            self.img_paths.append(str(img_dir+"/"+file_))

    def __len__(self):
        return len(self.img_paths)
 
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image, label_image = generate_text_on_image_and_pixel_mask_from_path(img_path, self.x_size, self.y_size, self.n_channels, RNGseed=self.RNGseed)

        # image = ToTensor()(text_img)
        # label_image = ToTensor()(mask)
        # label_image = label_image.squeeze().flatten()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label_image = self.target_transform(label_image)

        image = image.transpose(-1, 0, 1)
        image = image.astype(np.float32)


        label_image = label_image.reshape((1,label_image.shape[0],label_image.shape[1]))

        return {"image":image, "mask":label_image}

class PILImageDataset(Dataset):
    #Transform options: 'random_crop', TBA 'detect-resize'
    def __init__(self, img_dir, x_size, y_size, n_channels=3, transform=None, RNGseed=None, target_transform=None):
        self.x_size = x_size
        self.y_size = y_size
        self.n_channels = n_channels
        self.img_dir = img_dir
        self.transform = transform 
        self.target_transform = target_transform
        self.img_paths = []
        self.RNGseed = RNGseed
        self.local_RNG = np.random.default_rng(RNGseed if RNGseed else 0)
        self.PRINTABLE_CHARACTERS = list(string.ascii_letters + string.digits + string.punctuation)
        self.FONTS = ["arial.ttf", "impact.ttf", "comic.ttf"]#["arial.ttf", "ariali.ttf", "arialbd.ttf", "arialbi.ttf",  "calibri.ttf", "calibrii.ttf", "calibrib.ttf", "calibriz.ttf", "cambriai.ttf", "cambriab.ttf", "cambriaz.ttf", "comic.ttf", "comici.ttf", "comicbd.ttf", "comicz.ttf", "consola.ttf", "consolai.ttf", "consolab.ttf", "consolaz.ttf", "constan.ttf", "constani.ttf", "constanb.ttf", "constanz.ttf", "corbell.ttf", "corbelli.ttf", "corbel.ttf", "corbeli.ttf", "corbelb.ttf", "corbelz.ttf", "cour.ttf", "couri.ttf", "courbd.ttf", "courbi.ttf", "ebrima.ttf", "ebrimabd.ttf", "framd.ttf", "framdit.ttf", "gadugi.ttf", "gadugib.ttf", "georgia.ttf", "georgiai.ttf", "georgiab.ttf", "georgiaz.ttf", "holomdl2.ttf", "impact.ttf", "inkfree.ttf", "javatext.ttf", "leelawui.ttf", "lucon.ttf", "l_10646.ttf", "malgun.ttf", "malgunbd.ttf", "malgunsl.ttf", "marlett.ttf", "himalaya.ttf", "ntailu.ttf", "ntailub.ttf", "phagspa.ttf", "phagspab.ttf", "micross.ttf", "taile.ttf", "taileb.ttf",  "mvboli.ttf", "mmrtext.ttf", "mmrtextb.ttf", "pala.ttf", "palai.ttf", "palab.ttf", "palabi.ttf", "segmdl2.ttf", "segoepr.ttf", "segoeprb.ttf", "segoesc.ttf", "segoescb.ttf", "segoeuil.ttf", "seguili.ttf", "segoeuisl.ttf", "seguisli.ttf", "segoeui.ttf", "segoeuii.ttf", "seguisb.ttf", "seguisbi.ttf", "segoeuib.ttf", "segoeuiz.ttf", "seguibl.ttf", "seguibli.ttf", "seguihis.ttf", "seguiemj.ttf", "seguisym.ttf", "simsunb.ttf", "sylfaen.ttf", "symbol.ttf", "tahoma.ttf", "tahomabd.ttf", "times.ttf", "timesi.ttf", "timesbd.ttf", "timesbi.ttf", "trebuc.ttf", "trebucit.ttf", "trebucbd.ttf", "trebucbi.ttf", "verdana.ttf", "verdanai.ttf", "verdanab.ttf", "verdanaz.ttf", "webdings.ttf", "wingding.ttf"]    
        for file_ in  os.listdir(img_dir+"/"):
            self.img_paths.append(str(img_dir+"/"+file_))

    def put_text_and_mask_image_text_pixels_only(self, img):

        width, height = img.size

        topLeftCornerOfText = (int(self.local_RNG.integers(0, width*0.5, 1)), int(self.local_RNG.integers(20, height, 1))) #randomly select a place on the image
        text = ' '.join(self.local_RNG.choice(self.PRINTABLE_CHARACTERS, size=self.local_RNG.integers(0,50,1), shuffle=False)) #randomly create a string of printable characters, between 1 and 50 characters in length
        font_name = self.FONTS[int(self.local_RNG.integers(0, len(self.FONTS), 1))] #randomly select a font from the FONTS tuple
        base_font_size = int(self.local_RNG.integers(11, 120, 1))
        fontColor = (255-self.local_RNG.integers(0,10),255-self.local_RNG.integers(0,10),255-self.local_RNG.integers(0,10))#(int(255*(self.local_RNG.random())),int(255*(self.local_RNG.random())),int(255*(self.local_RNG.random()))) #randomly select a color, weighted towards darker colors by cubing the random number [0,1] //TODO change these values to be more realistic
        thickness = int((0.08)*base_font_size+0.8)
        try:
            font = ImageFont.truetype(font_name, base_font_size)
        except Exception as e:
            font = ImageFont.truetype("impact.ttf", base_font_size)
            print("exception:", e, "on font", font_name)

        draw = ImageDraw.Draw(img)
        mask = Image.new(mode="1", size=img.size)
        draw_mask = ImageDraw.Draw(mask)
        if self.local_RNG.integers(0,1,1) == 0:
            # draw.text(topLeftCornerOfText,text,(0),font=font, stroke_width=thickness*4)
            draw.text(topLeftCornerOfText,text,fontColor,font=font, stroke_width=thickness, stroke_fill=(0))
            draw_mask.text(topLeftCornerOfText,text,1,font=font, stroke_width=thickness)
        else:
            draw.text(topLeftCornerOfText,text,fontColor,font=font, stroke_width=thickness)
            draw_mask.text(topLeftCornerOfText,text,1,font=font, stroke_width=thickness)         

        return img, mask

    #generate random crop of image then put text and mask
    def generate_text_on_image_and_pixel_mask_from_path(self, path, x_size, y_size, n_channels):
        assert n_channels == 3, "Only n_channels == 3 supported"

        raw_img = None
        if type(path)==type(""):
            raw_img = Image.open(path)
            raw_img = raw_img.convert("RGB")
        else:
            print("path not string, but is", type(path))

        if(raw_img is not None):
            # cv.imshow('img',text_img)
            # cv.imshow('mask',mask)

            width, height = raw_img.size

            if width > x_size or height > y_size:
                raw_img = raw_img.crop((0,0,min(width, x_size), min(height, y_size)))

            padded_img = Image.new(raw_img.mode, (x_size, y_size))
            padded_img.paste(raw_img, (0,0))

            #TODO make sure that the text gets placed on the image properly
            text_img, mask = self.put_text_and_mask_image_text_pixels_only(padded_img)

            return text_img, mask

    def __len__(self):
        return len(self.img_paths)
 
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image, label_image = self.generate_text_on_image_and_pixel_mask_from_path(img_path, self.x_size, self.y_size, self.n_channels)
        
        image = np.array(image)
        label_image = np.array(label_image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label_image = self.target_transform(label_image)

        image = image.transpose(-1, 0, 1)
        image = image.astype(np.float32)

        label_image = label_image.astype(np.int8)
        label_image[label_image==True] = 1
        label_image[label_image==False] = 0

        label_image = label_image.reshape((1,label_image.shape[0],label_image.shape[1]))

        return {"image":image, "mask":label_image}


       
class MemeImageDataset(Dataset):
    #Transform options: 'random_crop', TBA 'detect-resize'
    def __init__(self, img_dir, transform=None, RNGseed=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform 
        self.target_transform = target_transform
        self.img_paths = []
        self.RNGseed = RNGseed
        for file_ in  os.listdir(img_dir+"/"):
            self.img_paths.append(str(img_dir+"/"+file_))

    def __len__(self):
        return len(self.img_paths)

    def read_meme(self, path):

        raw_img = None
        if type(path)==type(""):
            raw_img = cv.imread(path, flags=cv.IMREAD_COLOR) #TODO change this for ALPHA channel support
            # __log_("raw_image", raw_img)
        else:
            print("path not string, but is", type(path))

        if(raw_img is not None):
            padded_image = np.pad(raw_img, ((0,32-(raw_img.shape[0])%32), (0,32-(raw_img.shape[1])%32), (0,0)), 'constant', constant_values=(0))
            return padded_image
 
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.read_meme(img_path)

        if self.transform:
            image = self.transform(image)

        image = image.transpose(-1, 0, 1)
        image = image.astype(np.float32)

        return image

    
