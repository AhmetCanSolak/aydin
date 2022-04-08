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
    newyork,
    lizard,
    pollen,
    characters,
)
from aydin.it.classic_denoisers.spectral import calibrate_denoise_spectral
from aydin.util.log.log import Log


def demo_spectral(image, display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    function, parameters, memreq = calibrate_denoise_spectral(noisy)
    denoised = function(noisy, **parameters)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("         noisy   :", psnr_noisy, ssim_noisy)
    print("spectral denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')
        napari.run()

    return ssim_denoised


if __name__ == "__main__":
    newyork_image = newyork()
    demo_spectral(newyork_image)
    characters_image = characters()
    demo_spectral(characters_image)
    pollen_image = pollen()
    demo_spectral(pollen_image)
    lizard_image = lizard()
    demo_spectral(lizard_image)
    dots_image = dots()
    demo_spectral(dots_image)
    camera_image = camera()
    demo_spectral(camera_image)
