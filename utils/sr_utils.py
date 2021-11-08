from .common_utils import *

def load_LR_HR_imgs_sr(fname, imsize, dfactor, enforse_div32=None):
    """Loads an image, resizes it, center crops and downscales.

    Args:
        fname: path to the image
        imsize: new size for the img, -1 for no resizing
        dfactor: downscaling factor
        enforse_div32: Ensure the image's size is divisible by 32
    """

    # Load the image, resize it, and get the img of np type.
    img_orig_pil, img_orig_np = get_image(fname, imsize)

    # To ensure img'size are divisible by 32
    if enforse_div32 == 'CROP':
        new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32,
                    img_orig_pil.size[1] - img_orig_pil.size[1] % 32)

        bbox = [
            (img_orig.pil.size[0] - new_size[0])/2,
            (img_orig.pil.size[1] - new_size[1])/2,
            (img_orig.pil.size[0] + new_size[0])/2,
            (img_orig.pil.size[1] + new_size[1])/2
        ]

        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)
    else:
        img_HR_pil, img_HR_np = img_orig_pil, img_orig_np

    LR_size = [
        img_HR_pil.size[0] // factor,
        img_HR_pil.size[1] // factor
    ]

    img_LR_pil = img_HR_pil.resize(LR_size, Image)
    img_LR_np = pil_to_np(img_LR_pil)

    print('HR and LR resolutions: {}, {}'.format(str(img_HR_pil.size), str(img_LR_pil.size)))

    return{
        'orig_pil': img_orig_pil,
        'orig_np': img_orig_np,
        'LR_pil': img_LR_pil,
        'LR_np': img_LR_np,
        'HR_pil': img_HR_pil,
        'HR_np': img_HR_np
    }


def get_baselines(img_LR_pil, img_HR_pil):
    """Gets 'bicubic', 'sharpend bicubic' and 'nearest' baselines"""

    img_bicubic_pil = img_LR_pil.resize(img_HR_pil.size, Image.BICUBIC)
    img_bicubic_np = pil_to_np(img_bicubic_pil)

    img_nearest_pil = img_LR_pil.resize(img_HR_pil.size, Image.NEAREST)
    img_nearest_np = pil_to_np(img_nearest_pil)

    img_bic_sharp_pil = img_bicubic_pil.filter(PIL.ImageFilter.UnsharpMask())
    img_bic_sharp_np = pil_to_np(img_bic_sharp_pil)

    return img_bicubic_np, img_bic_sharp_np, img_nearest_np



def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])

    bbox = [
        int((target_size[0] - img_np.shape[1]) / 2),
        int((target_size[1] - img_np.shape[2]) / 2),
        int((target_size[0] + img_np.shape[1]) / 2),
        int((target_size[1] + img_np.shape[2]) / 2)
    ]

    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np

    return img_out
