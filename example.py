import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main():
    tflib.init_tf()
    _G, _D, Gs = pickle.load(open("results/00004-sgan-ffhq1024-3gpu/network-snapshot-008700.pkl", "rb"))
    Gs.print_layers()

    for i in range(0,1000):
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt)
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'test/example-'+str(i)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        print("Synthesis the image %s" % png_filename)

if __name__ == "__main__":
    main()

