import pickle

from frame import Frame
from manager import Manager

import os


def load_pkl_data(path):
    with open(path, 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')

    return data


class Controller:

    def __init__(self, images_path, pkl_path):
        self.images = self.read_images_path(images_path)
        self.file_pkl = pkl_path

    @staticmethod
    def read_images_path(basic_path):
        path_list = []

        for image in os.listdir(basic_path):
            path_list.append(basic_path + '/' + image)

        return path_list

    def run(self):
        current_frame = Frame()
        prev_frame = Frame()

        manage = Manager(load_pkl_data(self.file_pkl))

        for index, image in enumerate(self.images):
            current_frame.image_path = image

            prev_frame, current_frame = manage.run_frame(prev_frame, current_frame, index)


def main():
    controller = Controller("../data", "../Phase3/dusseldorf_000049.pkl")
    controller.run()


if __name__ == '__main__':
    main()
