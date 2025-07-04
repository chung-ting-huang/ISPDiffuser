# ISPDiffuser
This is the official employment of "ISPDiffuser: Learning RAW-to-sRGB Mappings with Texture-Aware Diffusion Models and Histogram-Guided Color Consistency"
# Inferece
1. Download the checkpoint of ZRR dataset and MAI dataset from [Baidu Netdisk](https://pan.baidu.com/s/1eAAZf0TbHakxrqwyl2AZVA?pwd=f1gv) or [GoogleDrive](https://drive.google.com/drive/folders/1smqKEttfKNEZfS8OT-g9yoMOCoRs90Gy?usp=drive_link), and put them into `ckpt` floder.

2. Download ZRR dataset and MAI dataset. MAI dataset can be download in [Baidu Netdisk](https://pan.baidu.com/s/1090u0vpmwD8swcop5BIWpA?pwd=wcib) or [GoogleDrive](https://drive.google.com/file/d/19XC0CtZ5WZsZPuuqsYVjq1z4TI7M08bt/view?usp=drive_link).
3. Change the dataset dir in `configs/MAI_dataset.yml` and `configs/ZRR_dataset.yml`
4. Run
   ```python evaluate.py --config path_to_ISPDiffuser/configs/MAI_dataset.yml```

資料格式讀取 txt檔案
requiment檔案
patch_size 設定問題
win_size 問題
ssim_gt = structural_similarity(recon_img, y, win_size=7, data_range=255, multichannel=True, gaussian_weights=True)
ISPDiffuser img_id 問題
