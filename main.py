import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import RandomState
from PIL import Image

from degradation import ImageDegradation
from ufpdeblur import UFPNet_DeblurringOnly


def main():
    import os
    from glob import glob
    from tqdm import tqdm

    device = "cuda"
    TRAIN_IMAGE_DIR = "data"
    TEST_IMAGE_DIR = "data/user27"

    rd = RandomState(42)
    
    def npimg2torch(x):
        return torch.from_numpy(x).unsqueeze(2).permute(2, 0, 1).unsqueeze(0).to(device)
    
    def torch2npimg(x):
        return x.detach().cpu().squeeze(0).permute(1, 2, 0).squeeze(2).numpy()
    
    degradation_model = ImageDegradation(rd, min_defocus_kernel = (4, 5), max_defocus_kernel = (12, 13), noise_poisson = None, noise_gaussian = None)

    model: UFPNet_DeblurringOnly = UFPNet_DeblurringOnly().to(device)
    kernel_size = 27
    
    def test_model(model, logpath):
        kernel_temp = torch.zeros((1, 346 * 260, kernel_size, kernel_size))
        kernel_temp[:, :, kernel_size // 2, kernel_size // 2] = 1

        files = glob(os.path.join(TEST_IMAGE_DIR, "**/*.png"), recursive=True)

        pbar = tqdm(files[:100])
        i = 0
        n = 0
        loss_sum = 0
        os.makedirs(logpath, exist_ok=True)
        with torch.no_grad():
            for file in pbar:
                sharp = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                sharp_pad = cv2.copyMakeBorder(np.array(sharp), 13, 13, 13, 13, cv2.BORDER_REPLICATE)
                sharp = sharp.astype(np.float32) / 255
                sharp_pad = sharp_pad.astype(np.float32) / 255

                blurred = degradation_model.apply(sharp_pad)
                blurred = np.clip(blurred, 0, 1)

                sharp_torch = npimg2torch(sharp)
                blurred_torch = npimg2torch(blurred)

                sharp_est_torch = model(blurred_torch, kernel_temp)
                loss = F.mse_loss(sharp_est_torch, sharp_torch)

                loss_sum += loss.item()
                n += 1

                if i < 10:
                    Image.fromarray((torch2npimg(blurred_torch) * 255).astype(np.uint8)).save(os.path.join(logpath, f"{str(i).zfill(3)}_b.png"))
                    Image.fromarray((torch2npimg(sharp_torch) * 255).astype(np.uint8)).save(os.path.join(logpath, f"{str(i).zfill(3)}_gt.png"))
                    Image.fromarray((torch2npimg(sharp_est_torch) * 255).astype(np.uint8)).save(os.path.join(logpath, f"{str(i).zfill(3)}_est.png"))

                i += 1

        loss_avg = loss_sum / n
        print(f"avg loss: {loss_avg:.3e}")


    if False:
        model.load_state_dict(torch.load("logs//test.pth"))
        test_model(model, "logs/test")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        kernel_temp = torch.zeros((1, 346 * 260, kernel_size, kernel_size))
        kernel_temp[:, :, kernel_size // 2, kernel_size // 2] = 1

        files = glob(os.path.join(TRAIN_IMAGE_DIR, "**/*.png"), recursive=True)

        pbar = tqdm(files[:160000] * 10)
        i = 0
        for file in pbar:
            sharp = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            sharp_pad = cv2.copyMakeBorder(np.array(sharp), 13, 13, 13, 13, cv2.BORDER_REPLICATE)
            sharp = sharp.astype(np.float32) / 255
            sharp_pad = sharp_pad.astype(np.float32) / 255

            blurred = degradation_model.apply(sharp_pad)
            blurred = np.clip(blurred, 0, 1)

            sharp_torch = npimg2torch(sharp)
            blurred_torch = npimg2torch(blurred)

            sharp_est_torch = model(blurred_torch, kernel_temp)
            loss = F.mse_loss(sharp_est_torch, sharp_torch)

            pbar.set_description(f"loss: {loss.item():.3e}")

            if i % 20000 == 0:
                logdir = f"logs/{str(i).zfill(6)}"
                os.makedirs(logdir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(logdir, "net.pth"))
                test_model(model, logdir)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1

        print("Done")


if __name__ == "__main__":
    main()