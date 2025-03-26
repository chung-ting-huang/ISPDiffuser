import colour_demosaicing
import numpy as np
import torch
import os
from PIL import Image

import cv2
import numpy as np
import colour_demosaicing
from PIL import Image


def remove_black_level(img, black_level=63, white_level=4*255):
    img = np.maximum(img-black_level, 0) / (white_level-black_level)
    return img


def demosaic_process(raw):

    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    ch_G = ch_Gb / 2 + ch_Gr / 2
    RAW_combined = np.dstack((ch_R, ch_G, ch_B))
    RAW_norm = RAW_combined.astype(np.float32)

    return RAW_norm



# def get_raw_demosaic(raw, pattern='RGGB'):  # HxW
#     raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, pattern=pattern)
#     raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32))
#     return raw_demosaic  # 3xHxW



def range_compressor(x):
    return (np.log(1 + 50 * x)) / np.log(1 + 50)
def durandanddorsy(img, c=50):
    height, width = np.shape(img)[:2]
    img = img/img.max()
    epsilon = 0.000001
    # L = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2] + epsilon #Y - grey
    L = 1 / 61 * (20 * img[:, :, 0] + 40 * img[:, :, 1] + img[:, :, 2] + epsilon) 
    R = img[:, :, 0] / L
    G = img[:, :, 1] / L
    B = img[:, :, 2] / L
 
    L_log = np.log(L) 
 
    base_log = cv2.bilateralFilter(L_log, 9, 0.3, 0.3)
    detail_log = L_log - base_log
    compressionFactor = np.log(c) / (base_log.max() - base_log.min())
    log_abs_scale = base_log.max() * compressionFactor
    # Ld_log = base_log * compressionFactor + detail_log
    Ld_log = base_log * compressionFactor + detail_log - log_abs_scale
    
    out = np.zeros(img.shape)
    out[:,:,0] = R * np.exp(Ld_log)
    out[:,:,1] = G * np.exp(Ld_log)
    out[:,:,2] = B * np.exp(Ld_log)
    
    outt = np.zeros(out.shape, dtype=np.float32)
    cv2.normalize(out, outt, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
 
    return outt


def remove_black_level(img, black_lv=200, white_lv=4095):
    img = np.maximum(img.astype(np.float32)-black_lv, 0) / (white_lv-black_lv)
    return img
def pack_process(raw):

    # Reshape the input bayer image
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[0::2, 1::2]
    ch_Gb = raw[1::2, 0::2]
    ch_B = raw[1::2, 1::2]
    RAW_combined = np.dstack((ch_R, ch_Gr, ch_Gb,ch_B))

    return RAW_combined

def get_raw_demosaic(raw, pattern='RGGB'):  # HxW
    # raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, pattern=pattern)
    raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_DDFAPD(raw, pattern=pattern)
    # raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(raw,pattern=pattern)
    raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32))
    return raw_demosaic  # 3xHxW


def unpack_process(dema_img, raw_h, raw_w):
    raw = np.zeros((raw_h, raw_w))
    raw[0::2, 0::2]=dema_img[:,:,0]
    raw[0::2, 1::2]=dema_img[:,:,1]
    raw[1::2, 0::2]=  dema_img[:,:,2]
    raw[1::2, 1::2]=dema_img[:,:,3]
    return raw

def myresize(raw, h, w):
    raw_resize = np.zeros((h,w,4))
    raw_resize[0::3,0::3,:] = raw[0::4,0::4,:]
    raw_resize[2::3,2::3,:] = raw[3::4,0::4,:]
    raw_resize[1::3,1::3,:] =( raw[1::4, 1::4,:]+ raw[2::4, 2::4,:])/2
    return raw_resize

# def demosaic_raw(raw):
#     red, green, blue = apply_bayer_filter(raw)
#     reconstructed_image, intermediate = demosaicing_algorithm(
#         red + green + blue, save_intermediate=True)
#     return reconstructed_image

def tone_mapping(img):
    # 对图像进行对数色调映射，模拟 HDR -> LDR 的过程
    img_log = np.log1p(img)  # log(x+1) 可以压缩动态范围
    img_mapped = np.expm1(img_log)  # 再进行指数映射恢复图像
    img_mapped = np.clip(img_mapped, 0.0, 255.0).astype(np.uint8)  # 将像素值限制在有效范围
    return img_mapped


# Gamma 校正：根据显示设备的特性进行 gamma 校正
def gamma_correction(img, gamma=2.2):
    inv_gamma = 1.0 / gamma
    img_corrected = np.power(img / 255.0, inv_gamma) * 255.0
    img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)
    return img_corrected

def ISP_afterde(raw_de):
    wb = [0.4691,1,0.6614]
    img_wb = np.zeros_like(raw_de)
    img_wb[:,:,0] = raw_de[:,:,0]/wb[0]
    img_wb[:,:,1] = raw_de[:,:,1]/wb[1]
    img_wb[:,:,2] = raw_de[:,:,2]/wb[2]
    # img_wb = np.asarray(img_wb, dtype='uint16')
    # img_de = cv2.cvtColor(img_wb, cv2.COLOR_BAYER_RGGB2RGB)
    # img_de = np.asarray(img_de,dtype=np.float32)
    # img_de = img_de / (16200-2047)
    # img_de = get_raw_demosaic(img_wb)
    # img_de = np.zeros_like(img_de_1)
    # img_de[:,:,0] = img_de_1[:,:,2]
    # img_de[:,:,1] = img_de_1[:,:,1]
    # img_de[:,:,2] = img_de_1[:,:,0]
    # img_de = get_raw_demosaic(img_wb)

    ##CCM
    # srgb2xyz = np.mat([[0.4124564, 0.3575761, 0.1804375],
    #                     [0.2126729, 0.7151522,0.0721750],
    #                     [0.0193339,0.1191920,0.9503041]])
    # # cam2xyz = np.mat([[0.7188,0.1641,0.0781],
    # #                     [0.2656,0.8984, -0.1562],
    # #                     [0.0625,-0.4062,1.1719]])
    # cam2xyz = np.mat([[0.7188,0.1641,0.0781],
    #                 [0.2656,0.8984, -0.1562],
    #                 [0.0625,-0.4062,1.1719]])
    # img_de = img_wb
    # cam2xyz_norm = cam2xyz / np.repeat(np.sum(cam2xyz,1),3).reshape(3,3)
    # img_ccm = np.zeros_like(img_de)
    # img_ccm[:,:,0] = cam2xyz_norm[0,0]*img_de[:,:,0]+cam2xyz_norm[0,1]*img_de[:,:,1]+cam2xyz_norm[0,2]*img_de[:,:,2]
    # img_ccm[:,:,1] = cam2xyz_norm[1,0]*img_de[:,:,0]+cam2xyz_norm[1,1]*img_de[:,:,1]+cam2xyz_norm[1,2]*img_de[:,:,2]
    # img_ccm[:,:,2] = cam2xyz_norm[2,0]*img_de[:,:,0]+cam2xyz_norm[2,1]*img_de[:,:,1]+cam2xyz_norm[2,2]*img_de[:,:,2]

    ##Tone mapping
    img_tonemap = tone_mapping(img_wb)
    # img_tonemap = img_ccm
    ## final ccm
    # xyz2srgb = srgb2xyz.I
    # # cam2rgb = cam2xyz *xyz2srgb
    # xyz2srgb_norm = xyz2srgb / np.repeat(np.sum(xyz2srgb,1),3).reshape(3,3)
    # img_fina_ccm = np.zeros_like(img_tonemap)
    # img_fina_ccm[:,:,0] = xyz2srgb_norm[0,0]*img_tonemap[:,:,0]+xyz2srgb_norm[0,1]*img_tonemap[:,:,1]+xyz2srgb_norm[0,2]*img_tonemap[:,:,2]
    # img_fina_ccm[:,:,1] = xyz2srgb_norm[1,0]*img_tonemap[:,:,0]+xyz2srgb_norm[1,1]*img_tonemap[:,:,1]+xyz2srgb_norm[1,2]*img_tonemap[:,:,2]
    # img_fina_ccm[:,:,2] = xyz2srgb_norm[2,0]*img_tonemap[:,:,0]+xyz2srgb_norm[2,1]*img_tonemap[:,:,1]+xyz2srgb_norm[2,2]*img_tonemap[:,:,2]

    # ##Gamma
    # img_gamma = np.clip(img_fina_ccm,0,1)
    # img_gamma = np.power(img_gamma,(3))

    
    # 应用 gamma 校正
    final_image = gamma_correction(img_tonemap)
    return img_wb

def ISP(raw_image):
    #ISP
    # img_nor = remove_black_level(raw_image, black_lv=252,white_lv=4095)
    img_nor = raw_image
    # img_wb = np.asarray(img_wb, dtype='uint16')
    # img_de = cv2.cvtColor(img_wb, cv2.COLOR_BAYER_RGGB2RGB)
    # img_de = np.asarray(img_de,dtype=np.float32)
    # img_de = img_de / (16200-2047)
    # img_de = get_raw_demosaic(img_wb)
    # img_de = np.zeros_like(img_de_1)
    # img_de[:,:,0] = img_de_1[:,:,2]
    # img_de[:,:,1] = img_de_1[:,:,1]
    # img_de[:,:,2] = img_de_1[:,:,0]
    img_de = get_raw_demosaic(img_nor)

    ##wb
    wb = [0.53,1,0.6614]
    img_wb = np.zeros_like(img_de)
    img_wb[:,:,0] = img_de[:,:,0]/wb[0]
    img_wb[:,:,1] = img_de[:,:,1]/wb[1]
    img_wb[:,:,2] = img_de[:,:,2]/wb[2]

    ##CCM
    srgb2xyz = np.mat([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522,0.0721750],
                        [0.0193339,0.1191920,0.9503041]])
    # cam2xyz = np.mat([[0.7188,0.1641,0.0781],
    #                     [0.2656,0.8984, -0.1562],
    #                     [0.0625,-0.4062,1.1719]])
    cam2xyz = np.mat([[0.7188,0.1641,0.0781],
                    [0.2656,0.8984, -0.1562],
                    [0.0625,-0.4062,1.1719]])
    

    cam2xyz_norm = cam2xyz / np.repeat(np.sum(cam2xyz,1),3).reshape(3,3)
    img_ccm = np.zeros_like(img_wb)
    img_ccm[:,:,0] = cam2xyz_norm[0,0]*img_wb[:,:,0]+cam2xyz_norm[0,1]*img_wb[:,:,1]+cam2xyz_norm[0,2]*img_wb[:,:,2]
    img_ccm[:,:,1] = cam2xyz_norm[1,0]*img_wb[:,:,0]+cam2xyz_norm[1,1]*img_wb[:,:,1]+cam2xyz_norm[1,2]*img_wb[:,:,2]
    img_ccm[:,:,2] = cam2xyz_norm[2,0]*img_wb[:,:,0]+cam2xyz_norm[2,1]*img_wb[:,:,1]+cam2xyz_norm[2,2]*img_wb[:,:,2]

    ##Tone mapping

    # img_tonemap = img_ccm
    ## final ccm
    xyz2srgb = srgb2xyz.I
    xyz2srgb_norm = xyz2srgb / np.repeat(np.sum(xyz2srgb,1),3).reshape(3,3)
    img_fina_ccm = np.zeros_like(img_wb)
    img_fina_ccm[:,:,0] = xyz2srgb_norm[0,0]*img_ccm[:,:,0]+xyz2srgb_norm[0,1]*img_ccm[:,:,1]+xyz2srgb_norm[0,2]*img_ccm[:,:,2]
    img_fina_ccm[:,:,1] = xyz2srgb_norm[1,0]*img_ccm[:,:,0]+xyz2srgb_norm[1,1]*img_ccm[:,:,1]+xyz2srgb_norm[1,2]*img_ccm[:,:,2]
    img_fina_ccm[:,:,2] = xyz2srgb_norm[2,0]*img_ccm[:,:,0]+xyz2srgb_norm[2,1]*img_ccm[:,:,1]+xyz2srgb_norm[2,2]*img_ccm[:,:,2]
    img_tonemap = range_compressor(img_wb)
    # ##Gamma
    # img_gamma = np.clip(img_fina_ccm,0,1)
    # img_gamma = np.power(img_gamma,(3))
    return img_tonemap

def range_expanding(x):
    output = (np.exp(x*np.log(1+50))-1)/50
    output = np.ascontiguousarray(output.astype(np.float32))
    return output
def inverse_ISP(rgb):
    rgb = rgb / 255.0
    img_tonemap_inverse= range_expanding(rgb)

def ISP_process(raw_imgs):
    assert len(raw_imgs.shape) == 3, 'ISP should process the unpacked raw image, \
        but the input dims are :{}'.format(raw_imgs.shape)
    for raw_img in raw_imgs:
        raw_img = raw_img
        raw_img = raw_img.detach().cpu().numpy()
        de_img = ISP(raw_img)
        rgb_img = np.clip(de_img*255.0, 0.0, 255.0).astype('uint8')
        # img_save_path = os.path.join(config.training.img_save_path, str(config.data.scale))
        # save_img = Image.fromarray(rgb_img)
        # if not os.path.exists(img_save_path):
        #     os.makedirs(img_save_path)
        # save_img.save(os.path.join(img_save_path, '{}_{}_{}.jpg'.format(image_name,psnr,ssim)))
        return rgb_img