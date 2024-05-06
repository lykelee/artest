import numpy as np
from math import tanh
from numpy.random import RandomState
from scipy.signal import convolve
from typing import Tuple, Union

class ImageDegradation():
    def __init__(
        self,
        rd: RandomState,
        min_defocus_kernel: Tuple[int, int],
        max_defocus_kernel: Tuple[int, int],
        noise_poisson: Union[None, Tuple[float, float]],
        noise_gaussian: [None, Tuple[float, float]]):
        self._rd = rd
        self.min_defocus_kernel_size = min_defocus_kernel
        self.max_defocus_kernel_size = max_defocus_kernel
        self.noise_poisson = noise_poisson
        self.noise_gaussian = noise_gaussian
        self.defocus_kernel = [self._get_disk_kernel(kernel_size) for kernel_size in range(self.min_defocus_kernel_size[0], self.max_defocus_kernel_size[1] + 1)]

    def _inverse_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        return image ** 2.2
    
    def _get_disk_kernel(self, kernel_size: int, mode: str = "soft", sigma: float = 0.25, phi: float = 0.5) -> np.ndarray:
        if mode != "soft" and mode != "hard":
            raise Exception("Disk kernel mode should be either soft or hard, but given mode is {}".format(mode))
        kernel_size = kernel_size * 2 - 1
        
        kernel = np.zeros((kernel_size, kernel_size), dtype = np.float32)
        center = kernel_size // 2

        for r in range(kernel_size):
            for c in range(kernel_size):
                x, y = abs(r - center), abs(c - center)
                if mode == "hard":
                    kernel[r][c] = 0 if x * x + y * y > center * center else 1
                else:
                    kernel[r][c] = 0.5 + 0.5 * tanh(sigma * (center * center - x * x - y * y) + phi)

        kernel = kernel / np.sum(kernel)

        return kernel

    def _add_defocus(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        kernel_index = kernel_size - self.min_defocus_kernel_size[0]
        crop = self.max_defocus_kernel_size[1]
        result = convolve(image, self.defocus_kernel[kernel_index], "same")
        return result[crop:-crop, crop:-crop]

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        if self.noise_poisson is not None:
            beta1 = self._rd.uniform(*self.noise_poisson)
            image = image + self._rd.poisson((image / beta1)) * beta1
        if self.noise_gaussian is not None:
            beta2 = self._rd.uniform(*self.noise_gaussian)
            image = image + self._rd.uniform(0, 1, image.shape).astype(np.float32) * beta2

        return image
    
    def _add_lensblur(self, image: np.ndarray) -> np.ndarray:
        min_kernel_size = self._rd.randint(self.min_defocus_kernel_size[0], self.min_defocus_kernel_size[1] + 1)
        max_kernel_size = self._rd.randint(self.max_defocus_kernel_size[0], self.max_defocus_kernel_size[1] + 1)

        images = []
        for i in range(min_kernel_size, max_kernel_size + 1):
            images.append(self._add_defocus(image, i))
        images = np.stack(images, axis = 2)

        H, W, C = images.shape
        if C == 1:
            return images[:, :, 0]
        center_H = H // 2
        center_W = W // 2

        H_axis = np.zeros((H, W)).astype(np.int32)
        W_axis = np.zeros((H, W)).astype(np.int32)
        for i in range(H):
            H_axis[i, :] = i
        for i in range(W):
            W_axis[:, i] = i
        index = (H_axis - center_H) ** 2 + (W_axis - center_W) ** 2
        A = 5
        index = (index - np.min(index)) / (np.max(index) - np.min(index))
        index = (np.sqrt(index + A * A) - A) / (np.sqrt(1 + A * A) - A)
        index_lower = (index * (C - 2)).astype(np.int32)
        index_higher = (index * (C - 2) + 1).astype(np.int32)
        index_weight = index_higher - index * (C - 2)

        result_lower = images[H_axis, W_axis, index_lower]
        result_higher = images[H_axis, W_axis, index_higher]
        result = result_lower * index_weight + result_higher * (1 - index_weight)

        return result

    def apply(self, image: np.ndarray) -> np.ndarray:
        # image = self._inverse_gamma_correction(image)
        image = self._add_lensblur(image)
        image = self._add_noise(image)
        image = np.clip(image, 0, 1)

        return image.astype(np.float32)

def main():
    from PIL import Image
    import cv2
    from tqdm import tqdm
    
    TEST_IMAGE_DIR = "data/user1/0/frames/1_0_0_st_34291754.png"

    img = Image.open(TEST_IMAGE_DIR).convert("L")
    img = np.array(img).astype(np.float32) / 255
    rd = RandomState(42)

    degradation_model = ImageDegradation(rd, min_defocus_kernel = (4, 5), max_defocus_kernel = (12, 13), noise_poisson = None, noise_gaussian = None)

    result = degradation_model.apply(img)
    result = cv2.resize(result, dsize = (0, 0), fx = 4, fy = 4, interpolation = cv2.INTER_LANCZOS4)
    result = np.clip(result, 0, 1)
    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save("input.png")

if __name__ == "__main__":
    main()