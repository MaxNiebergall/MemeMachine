
# Sources (adapted from):
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://github.com/shervinea/enzynet/blob/master/enzynet/volume.py#L24 -> MIT License


from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import keras
import numpy as np

import image_with_text_functions

class TextMaskImageDataGenerator(Sequence):
    """Generates batches of images with Text with associated pixel masks (class labels) on the fly.
    To be passed as argument in the fit_generator function of Keras.
    Parameters
    ----------
    x_size : int (optional, default is 256)
        Size of each side of the images.
    y_size : int (optional, default is 256)
        Size of each side of the images.
    n_channels : int (optional, default is 3) 
        Currently only supports 3 channels.
    batch_size : int (optional, default is 32)
        Number of samples in output array of each iteration of the 'generate'
        method.
    list_image_paths : list of strings FIXME
        List of enzymes to generate.
    shuffle : boolean (optional, default is True)
        If True, shuffles order of exploration.


    """
    def __init__(
            self,
            list_image_paths: List[Text],
            x_size: int = 256,
            y_size: int = 256,
            batch_size: int = 32,
            shuffle: bool = True,

    ) -> None:
        """Initialization."""
        self.batch_size = batch_size
        self.x_size = x_size
        self.y_size = y_size
        self.list_image_paths = list_image_paths
        self.n_channels = 3 #TODO add support for ALPHA channel
        self.shuffle = shuffle
        self.on_epoch_end()




    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_image_paths), dtype=np.int)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.list_image_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        # Generate indexes of the batch.
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] #TODO index must be i/batch_size?

        # Find list of IDs.
        list_image_paths_temp = [self.list_image_paths[k] for k in indexes]

        # Generate data.
        X, y = self.__data_generation(list_image_paths_temp)

        return X, y

    def __data_generation(
            self,
            list_image_ids_temp: List[np.array]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns images with text and pixel masks ."""  # X : (batch_size, x_size, y_size, n_channels).

        X = np.empty((self.batch_size,  # batch_size.
                      self.x_size,  # dimension w.r.t. x.
                      self.y_size,  # dimension w.r.t. y.
                      self.n_channels), dtype=np.uint8)  # n_channels.
        y = np.empty((self.batch_size,  # batch_size.
                      self.x_size,  # dimension w.r.t. x.
                      self.y_size,  # dimension w.r.t. y.
                      self.n_channels), dtype=np.uint8)  # n_channels.

        image_index=0
        batch_index=0
        while batch_index < self.batch_size:
            print("__data_gen, ", list_image_ids_temp[image_index])
            imgs, masks = image_with_text_functions.generate_crops_of_text_on_image_and_pixel_mask_from_path(list_image_ids_temp[image_index], self.x_size, self.y_size, self.n_channels)
            num_crops = min(len(imgs), self.batch_size-batch_index)
            if num_crops >0:
                # __log_("__data_generation.X",str(X[image_index:image_index+num_crops].shape))
                # __log_("__data_generation.imgs[:num_crops]",str(np.asarray(imgs[:num_crops]).shape))
                # print("o,gs", np.asarray(imgs[:num_crops]).shape)

                X[batch_index:batch_index+num_crops, 0:self.x_size, 0:self.y_size, 0:self.n_channels] = np.asarray(imgs[:num_crops])                
                y[batch_index:batch_index+num_crops, 0:self.x_size, 0:self.y_size, 0:self.n_channels] = np.asarray(masks[:num_crops])
                # makss = masks[:num_crops]
                # import cv2 as cv
                # for i in range(len(makss)):
                #     cv.imshow('makss'+str(i),makss[i])
                # for i in range(num_crops):
                #     cv.imshow('makss'+str(i), y[batch_index+i])
                # cv.waitKey(0)

                image_index+=num_crops
                batch_index+=num_crops

            image_index+=1

        print("data generated shapes", X.shape, y.shape)
        return X, y
