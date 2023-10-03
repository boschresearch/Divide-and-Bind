import torch
import numpy as np
import math
from PIL import Image
import cv2
from einops import rearrange
from IPython.display import display
from typing import Union, List


#####################################################
# For Cross-Attention
#####################################################
def reshape_CA_attn_map(attn_map, res_required=16):
    # input: [bs*h, L, c], L=h*w, c=77
    # output: [bs, H W c]
    res = int(math.sqrt(attn_map.shape[1]))
    attn_map = rearrange(attn_map, 'b (h w) c -> b c h w', h=res)
    if res != res_required:
        attn_map = torch.nn.functional.interpolate(attn_map, size=(res_required, res_required), mode='bilinear')
    attn_map = rearrange(attn_map, 'b c h w -> b h w c')
    return attn_map


def aggregate_attention(attention_map_list, res_required) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    for attn_map in attention_map_list:
        attn_map = reshape_CA_attn_map(attn_map, res_required=res_required)
        out.append(attn_map)
    out = torch.cat(out, dim=0)  # [40,16,16,77]
    return out

#####################################################
# Visualization helper
#####################################################
def min_max_norm(x, min_v=None, max_v=None):
    if min_v is not None and max_v is not None:
        y = (x - min_v) / (max_v - min_v)
    else:
        y = (x - x.min()) / (x.max() - x.min())
    return y

def show_image_relevance(image_relevance, image, relevnace_res=256, min_v=None, max_v=None):
    # image_relevance:  (h,w)
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        mask = mask.cpu()
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    if isinstance(image,np.ndarray):
        image = Image.fromarray(image)
    image = np.array(image.resize((relevnace_res, relevnace_res)))
    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res, mode='bilinear')
    image_relevance = min_max_norm(image_relevance, min_v=min_v, max_v=max_v)
    image_relevance = image_relevance.reshape(relevnace_res, relevnace_res)
    image = min_max_norm(image)
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        try:
            display(pil_img)
        except:
            pil_img.show()
    return pil_img