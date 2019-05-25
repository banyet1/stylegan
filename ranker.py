import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys

def main():
    tflib.init_tf()
    _G, D, _Gs = pickle.load(open(sys.argv[1], "rb"))
    image_filenames = sys.argv[2:]

    for i in range(0, len(image_filenames)):
        img = np.asarray(PIL.Image.open(image_filenames[i]))
        img = img.reshape(1, 3,512,512)
        score = D.run(img, None)
        print(image_filenames[i], score[0][0])

if __name__ == "__main__":
    main()
