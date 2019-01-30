import re
import os
import numpy as np
import config

from utils import sparse_tuple_from, resize_image, label_to_array

from scipy.misc import imsave

class DataManager(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_test_ratio, max_char_count):
        """
        max_image_width: the width that will be used in input placeholder. this is the upper limit for the width of the prepared image.
        """


        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        print(train_test_ratio)

        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.max_char_count = max_char_count
        self.data, self.data_len = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset
        self.train_batches = self.__generate_all_train_batches()
        self.test_batches = self.__generate_all_test_batches()

    def __load_data(self):
        """
            this function populates self.data
            Load all the images in the folder
        """

        print('Loading data')

        examples = []

        count = 0
        skipped = 0
        print("examples_path = {}".format(self.examples_path))
        example_files = os.listdir(self.examples_path)

        print("example files size:{}".format(len(example_files)))
        for f in os.listdir(self.examples_path):

            label_string = f.split('_')[0]

            if len(label_string) > self.max_char_count:
                continue

            # initial_len = col size
            # arr: [row,col] shaped np.array
            arr, initial_len = resize_image(
                os.path.join(self.examples_path, f),
                self.max_image_width
            )

            # arr: the array of image data
            # 

            label_char_index_list = label_to_array(label_string)

            examples.append(
                (
                    arr,
                    label_string,
                    label_char_index_list
                )
            )

            # debug 
            imsave('blah.png', arr)
            
            count += 1

        print("__load_data: len(examples)={}".format(len(examples)))

        return examples, len(examples)

    def __generate_all_train_batches(self):
        train_batches = []
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.data[old_offset:new_offset])
            # batch_la: batch of la(label array. sequence of char index)
            # batch_y: batch of the actual strings

            # converting label_string into a numpy array with shape=(1,) where the first element
            # contains the label_string
            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            # batch_dt: converted from char index array. in other words, this is an array of one-hot-vector
            # batch_dt is a tuple
            batch_dt = sparse_tuple_from(
                np.reshape(
                    np.array(raw_batch_la),
                    (-1)
                )
            )
            print("batch_dt:{}".format(batch_dt))

            # I think this swapping is changing [row,col] -> [col,row] for each input image. 
            # this way, we can feed each vertical slice of the image starting from the left into the model input.
            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            train_batches.append((batch_y, batch_dt, batch_x))
        return train_batches

    def __generate_all_test_batches(self):
        test_batches = []
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_dt = sparse_tuple_from(
                np.reshape(
                    np.array(raw_batch_la),
                    (-1)
                )
            )

            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            test_batches.append((batch_y, batch_dt, batch_x))
        return test_batches
