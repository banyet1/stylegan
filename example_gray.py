import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import six
import io
from PIL import Image

def main():
    tflib.init_tf()
    _G, _D, Gs = pickle.load(open("/root/styleganface/results/00009-sgan-ffhq1024-2gpu/network-snapshot-009640.pkl", "rb"))
    Gs.print_layers()

    for i in range(0,1000):
        array_list = []
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt)
        array_list.append(images)
        images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt)
        array_list.append(images)
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'test/example-'+str(i)+'.png')
        stream = io.BytesIO(images[0])
        img = Image.frombuffer('L', (1024, 1024), images[0])
        #img = Image.open(stream)
        img1 = img.transpose(Image.FLIP_TOP_BOTTOM)
        img1.save(png_filename)
        #PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        print("Synthesis image: %s (%d)" %(png_filename, i));

if __name__ == "__main__":
    main()

