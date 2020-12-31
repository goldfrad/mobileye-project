try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def convolution(color_imj):
    t = 32
    k = -32
    kernel = np.array([[k, k, k, k, k],
                       [k, t, t, t, k],
                       [k, t, 255, t, k],
                       [k, t, t, t, k],
                       [k, k, k, k, k]])

    grad = sg.convolve2d(color_imj, kernel, boundary='symm', mode='same')
    return ndimage.maximum_filter(grad, size=7)


def get_max_candidate(img):
    height, width = img.shape[:2]

    x_point = []
    y_point = []

    for x in range(10, width, 7):
        for y in range(10, height, 7):
            if img[y, x] > 22000:
                x_point.append(x)
                y_point.append(y)

    return x_point, y_point


def find_tfl_lights(c_image: np.ndarray):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    red_img = c_image[:, :, 0]
    green_img = c_image[:, :, 1]
    new_red = convolution(red_img)
    new_green = convolution(green_img)
    x_red, y_red = get_max_candidate(new_red)
    x_green, y_green = get_max_candidate(new_green)
    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=3)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=3)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running,
     because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
