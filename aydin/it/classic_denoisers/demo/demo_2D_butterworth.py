# flake8: noqa
import numpy
import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.io.datasets import (
    normalise,
    add_noise,
    dots,
    lizard,
    pollen,
    newyork,
    characters,
)
from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.log.log import Log


def demo_butterworth(image, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(7)

    image = normalise(image.astype(np.float32))

    # we add noise:
    noisy = add_noise(image)

    function, parameters, memreq = calibrate_denoise_butterworth(noisy)
    denoised = function(noisy, **parameters)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("        noisy   :", psnr_noisy, ssim_noisy)
    print("lowpass denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(noisy, name='noisy')
            viewer.add_image(denoised, name='denoised')

    return ssim_denoised


if __name__ == "__main__":
    demo_butterworth(pollen())
    demo_butterworth(dots())
    demo_butterworth(newyork())
    demo_butterworth(characters())
    demo_butterworth(lizard())
    demo_butterworth(camera())
