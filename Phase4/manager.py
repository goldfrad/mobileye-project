import copy

import numpy as np
from PIL import Image
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

from Phase1.find_red_and_green_lights import find_tfl_lights
from Phase3.SFM import calc_TFL_dist


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


def get_lights(img):
    image = np.array(Image.open(img))
    x_red, y_red, x_green, y_green = find_tfl_lights(image)
    points = []

    for x, y in zip(x_red, y_red):
        points.append([x, y])

    for x, y in zip(x_green, y_green):
        points.append([x, y])
    colors = ['r'] * len(x_red) + ['g'] * len(x_green)

    return points, colors


def network(images: np.ndarray, model):
    crop_shape = (81, 81)
    prediction = model.predict(images.reshape([-1] + list(crop_shape) + [3]))

    return prediction[0][1] > 0.5


def crop(candidate, image):
    height, width = image.shape[:2]

    x = candidate[0]
    y = candidate[1]

    if x < 40:
        x = 40

    elif x > height - 41:
        x = height - 41

    if y < 40:
        y = 40

    elif y > width - 41:
        y = width - 41
    im = image[x - 40:x + 41, y - 40:y + 41]

    return im


def get_dists(prev, current, focal, pp, em):
    prev_c = FrameContainer(prev.image_path)
    current_c = FrameContainer(current.image_path)
    prev_c.traffic_light = prev.tfl_candidates
    current_c.traffic_light = current.tfl_candidates
    current_c.EM = em
    curr_frame = calc_TFL_dist(prev_c, current_c, focal, pp)

    return np.array(curr_frame.traffic_lights_3d_location)[:, 2]


def visualizition(frame):
    fig, (lights, tfls, dists) = plt.subplots(1, 3, figsize=(12, 10))
    lights.imshow(plt.imread(frame.image_path))
    candidates = np.array(frame.light_candidates)
    x, y = candidates[:, 0], candidates[:, 1]
    lights.scatter(x, y, c=frame.colors_lights, s=1)
    lights.set_title('Source lights')

    tfls.imshow(plt.imread(frame.image_path))

    if frame.tfl_candidates != []:
        candidates = np.array(frame.tfl_candidates)
        x, y = candidates[:, 0], candidates[:, 1]
        tfls.scatter(x, y, c=frame.colors_tfl, s=1)
    tfls.set_title('Traffic lights lights')

    if frame.distances != list():
        x_cord, y_cord, image_dist = get_coords_tfl(frame)
        dists.imshow(plt.imread(frame.image_path))
        for i in range(len(x_cord)):
            dists.text(x_cord[i], y_cord[i], r'{0:.1f}'.format(frame.distances[i]), color='r')

    dists.set_title('Distances of tfl')
    plt.show()


def get_coords_tfl(frame):
    image = np.array(Image.open(frame.image_path))
    curr_p = frame.tfl_candidates
    x_cord = [p[0] for p in curr_p]
    y_cord = [p[1] for p in curr_p]

    return x_cord, y_cord, image


class Manager:
    def __init__(self, data):
        self.model = load_model("../Phase2/model.h5")
        self.pp = data['principle_point']
        self.focal = data['flx']
        self.list_EM = []
        self.data = data
        self.init_em()

    def run_frame(self, prev, current, i):
        # Phase1
        current.light_candidates, current.colors_lights = get_lights(current.image_path)
        image = np.array(Image.open(current.image_path))

        # Phase2
        for index, point in enumerate(current.light_candidates):
            crop_image = crop(point, image)

            if crop_image.any():
                isTFL = network(crop_image, self.model)

                if isTFL:
                    current.tfl_candidates.append(point)
                    current.colors_tfl.append(current.colors_lights[index])
        # Phase3
        print(prev.tfl_candidates)
        print(current.tfl_candidates)
        if prev.tfl_candidates != [] and current.tfl_candidates != []:
            current.distances = get_dists(prev, current, self.focal, self.pp, self.list_EM[i - 1])

        visualizition(current)
        prev = copy.deepcopy(current)

        return prev, current

    def init_em(self):
        for i in range(24, 29):
            EM = np.eye(4)
            EM = np.dot(self.data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            self.list_EM.append(EM)
