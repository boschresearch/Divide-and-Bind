import sys
sys.path.append(".")
sys.path.append("..")
import abc
import os
import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List
import math
from einops import rearrange
from diffusers.models.cross_attention import CrossAttention
from collections import defaultdict
import pickle
from .annotation_sd import aggregate_attention as agg_attn_func
from .annotation_sd import min_max_norm


def move_to(obj_to_move, device='cpu'):
    if isinstance(obj_to_move, dict):
        res = {}
        for k, v in obj_to_move.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj_to_move, list):
        res = []
        for v in obj_to_move:
            res.append(move_to(v, device))
        return res
    elif torch.is_tensor(obj_to_move):
        return obj_to_move.to(device)
    else:
        raise TypeError("Invalid type for move_to")


def save_to_pickle(obj, output_name):
    with open(output_name, 'wb') as handle:
        pickle.dump(obj, handle)


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = False) -> Image.Image:
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
        display(pil_img)
    return pil_img


def register_attention_control(model, controller):
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.processor = ControlCrossAttnProcessor(controller,place_in_unet) #ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count # 32


class ControlCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.time_counter = 0

    def get_attention_scores(self,attn_module, query, key, attention_mask=None):
        dtype = query.dtype
        if attn_module.upcast_attention:
            query = query.float()
            key = key.float()
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=attn_module.scale,
        )
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if attn_module.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
        del attention_scores
        return attention_probs

    def __call__(self, attn_module: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn_module.prepare_attention_mask(attention_mask, sequence_length)

        query = attn_module.to_q(hidden_states)
        query = attn_module.head_to_batch_dim(query)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn_module.to_k(encoder_hidden_states)
        value = attn_module.to_v(encoder_hidden_states)
        key = attn_module.head_to_batch_dim(key)
        value = attn_module.head_to_batch_dim(value)

        attention_map = self.get_attention_scores(attn_module, query, key, attention_mask)

        # store attention matrix to controller
        self.controller(attention_map, is_cross, self.place_in_unet)
        hidden_states = torch.bmm(attention_map, value)
        hidden_states = attn_module.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn_module.to_out[0](hidden_states)
        # dropout
        hidden_states = attn_module.to_out[1](hidden_states)

        return hidden_states


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead change to 32**2
            self.step_store[key].append(attn)
            #print(attn.shape) # [bs*h, h*w, h*w or 77]
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()


    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = []
        self.keep_timestep_dict = {}
        self.cur_time_step = 0
        self.curr_step_index = 0
        self.ca_bbox_list = []
        self.token_indices = []

    def set_time_step(self, t):
        self.cur_time_step = t

        if t in self.keep_timestep_list:
            cpu_store = move_to(self.attention_store, device='cpu')
            self.keep_timestep_dict[t] = cpu_store
            del cpu_store
        if self.store_every_step:
            cpu_store = move_to(self.attention_store, device='cpu')
            self.global_store.append(cpu_store)
            del cpu_store

    def save_attneion_map_to_pickle(self):
        if self.cur_time_step in self.pickle_store_timestep_list:
            file_name = f'{self.cur_time_step}.pickle'
            file_name = os.path.join(self.output_dir, file_name)
            print(file_name)
            save_to_pickle(self.attention_store,file_name)

    def cal_image_relevance(self,image_relevance, relevnace_res=256, min_v=None, max_v=None, mode='bilinear'):
        # image_relevance:  (h,w)
        # create heatmap from mask on image
        image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res, mode=mode)
        image_relevance = min_max_norm(image_relevance, min_v=min_v, max_v=max_v)
        image_relevance = image_relevance.reshape(relevnace_res, relevnace_res)
        heatmap = cv2.applyColorMap(np.uint8(255 * image_relevance), cv2.COLORMAP_JET)
        heatmap = np.uint8(heatmap)
        heatmap = cv2.cvtColor(np.array(heatmap), cv2.COLOR_RGB2BGR)
        return heatmap

    def vis_and_save_attn_map_nouse(self, pred_img, final_res_required=256,use_global_mixmax=True):
        cpu_store = move_to(self.attention_store, device='cpu')
        if cpu_store['down_cross'][4].shape[0] > 8:
            from_k = 8
        else:
            from_k = 0
        attn_list = [
            cpu_store['down_cross'][4][from_k:], cpu_store['down_cross'][5][from_k:],
            cpu_store['mid_cross'][0][from_k:], cpu_store['up_cross'][0][from_k:],
            cpu_store['up_cross'][1][from_k:], cpu_store['up_cross'][2][from_k:],
        ]
        res_required = 16
        agg_attn = agg_attn_func(attn_list, res_required=16)
        self.token_indices = [2,6]

        if use_global_mixmax:
            for i, selected_id in enumerate(self.token_indices):
                image_relevance = agg_attn[:, :, :, selected_id].mean(0)
                # image_relevance = agg_attn.mean(0)
                image_relevance = image_relevance * agg_attn[:, :, :, selected_id + 1].mean(0)
                if i == 0:
                    global_min = image_relevance.min()
                    global_max = image_relevance.max()
                else:
                    global_min = image_relevance.min() if image_relevance.min() < global_min else global_min
                    global_max = image_relevance.max() if image_relevance.max() > global_max else global_max
            print(global_max.item(), global_min.item())
        else:
            global_min = None
            global_max = None
        vis_list = []
        resized_img = cv2.resize(pred_img, dsize=(final_res_required, final_res_required), interpolation=cv2.INTER_CUBIC)
        vis_list.append(resized_img)
        os.makedirs(self.output_dir, exist_ok=True)
        for selected_id in self.token_indices:
            image_relevance = agg_attn[:, :, :, selected_id].mean(0)
            image_relevance = image_relevance * agg_attn[:, :, :, selected_id + 1].mean(0)
            vis_img = self.cal_image_relevance(
                image_relevance,  relevnace_res=final_res_required, min_v=global_min,max_v=global_max
            )

            vis_img = vis_img.astype(np.uint8)
            pil_img = Image.fromarray(vis_img)
            file_name = f'{self.cur_time_step}_{selected_id}.png'

            file_name = os.path.join(self.output_dir, file_name)
            print(file_name)
            pil_img.save(file_name)

    #_single
    def vis_and_save_attn_map_single(self, pred_img, final_res_required=256,use_global_mixmax=True):
        cpu_store = move_to(self.attention_store, device='cpu')
        if cpu_store['down_cross'][4].shape[0] > 8:
            from_k = 8
        else:
            from_k = 0
        attn_list = [
            cpu_store['down_cross'][4][from_k:], cpu_store['down_cross'][5][from_k:],
            cpu_store['mid_cross'][0][from_k:], cpu_store['up_cross'][0][from_k:],
            cpu_store['up_cross'][1][from_k:], cpu_store['up_cross'][2][from_k:],
        ]
        res_required = 16
        agg_attn = agg_attn_func(attn_list, res_required=16)

        if use_global_mixmax:
            for i, selected_id in enumerate(self.token_indices):
                image_relevance = agg_attn[:, :, :, selected_id].mean(0)
                if i == 0:
                    global_min = image_relevance.min()
                    global_max = image_relevance.max()
                else:
                    global_min = image_relevance.min() if image_relevance.min() < global_min else global_min
                    global_max = image_relevance.max() if image_relevance.max() > global_max else global_max
            print(global_max.item(), global_min.item())
        else:
            global_min = None
            global_max = None
        # !!! Change here !!!
        global_min = None
        global_max = None

        vis_list = []

        resized_img = cv2.resize(pred_img, dsize=(final_res_required, final_res_required), interpolation=cv2.INTER_NEAREST)
        vis_list.append(resized_img)
        for selected_id in self.token_indices:
            image_relevance = agg_attn[:, :, :, selected_id].mean(0)

            vis_img = self.cal_image_relevance(
                image_relevance,  relevnace_res=final_res_required, min_v=global_min,max_v=global_max,
                mode='nearest',
            )
            vis_img = vis_img.astype(np.uint8)
            file_name = f'{self.cur_time_step + 1}_{selected_id}.png'
            os.makedirs(self.output_dir, exist_ok=True)
            file_name = os.path.join(self.output_dir, file_name)
            print(file_name)
            pil_img = Image.fromarray(vis_img)
            pil_img.save(file_name)


    def vis_and_save_attn_map(self, pred_img, final_res_required=256,use_global_mixmax=True):
        cpu_store = move_to(self.attention_store, device='cpu')
        if cpu_store['down_cross'][4].shape[0] > 8:
            from_k = 8
        else:
            from_k = 0
        attn_list = [
            cpu_store['down_cross'][4][from_k:], cpu_store['down_cross'][5][from_k:],
            cpu_store['mid_cross'][0][from_k:], cpu_store['up_cross'][0][from_k:],
            cpu_store['up_cross'][1][from_k:], cpu_store['up_cross'][2][from_k:],
        ]
        res_required = 16
        agg_attn = agg_attn_func(attn_list, res_required=16)

        if use_global_mixmax:
            for i, selected_id in enumerate(self.token_indices):
                image_relevance = agg_attn[:, :, :, selected_id].mean(0)

                if i == 0:
                    global_min = image_relevance.min()
                    global_max = image_relevance.max()
                else:
                    global_min = image_relevance.min() if image_relevance.min() < global_min else global_min
                    global_max = image_relevance.max() if image_relevance.max() > global_max else global_max
            print(global_max.item(), global_min.item())
        else:
            global_min = None
            global_max = None
        # !!! Change here !!!
        global_min = None
        global_max = None

        vis_list = []
        resized_img = cv2.resize(pred_img, dsize=(final_res_required, final_res_required), interpolation=cv2.INTER_CUBIC)
        vis_list.append(resized_img)
        for selected_id in self.token_indices:
            image_relevance = agg_attn[:, :, :, selected_id].mean(0)

            vis_img = self.cal_image_relevance(
                image_relevance,  relevnace_res=final_res_required, min_v=global_min,max_v=global_max
            )
            vis_img = vis_img.astype(np.uint8)
            vis_list.append(vis_img)
        vis_pil = view_images(vis_list,num_rows=1,display_image=False)
        file_name = f'{self.cur_time_step + 1}.png'
        os.makedirs(self.output_dir, exist_ok=True)
        file_name = os.path.join(self.output_dir, file_name)
        print(file_name)
        vis_pil.save(file_name)
        #del vis_list

    def get_attn_dict(self):
        return self.keep_timestep_dict

    def set_token_indices(self, token_indices):
        self.token_indices = token_indices

    def __init__(self, save_global_store=False,
                 store_every_step=False, # Always False
                 output_dir=None,
                 pickle_store_timestep_list=None,
                 keep_timestep_list=None,
                 get_bbox_from_ca_at=25,
                 cal_bbox_from_ca=False,
                 save_attn_vis=False,
                 ):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store # no use!
        self.store_every_step = store_every_step
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = [] #defaultdict(list) #{}
        self.token_indices = []
        self.output_dir = output_dir # dir to save at every step
        self.save_attn_vis = save_attn_vis

       # only for saving attention map into pickle
        self.pickle_store_timestep_list = []
        if pickle_store_timestep_list is not None:
            self.pickle_store_timestep_list = pickle_store_timestep_list

        # only for saving attention map into CPU
        self.keep_timestep_list = []
        self.keep_timestep_dict = {}
        if keep_timestep_list is not None:
            self.keep_timestep_list = keep_timestep_list
        self.curr_step_index = 0
        self.cur_time_step = 0

        # If calculating bbox from CA on the fly
        self.ca_bbox_list = []
        self.get_bbox_from_ca_at = get_bbox_from_ca_at
        self.cal_bbox_from_ca = cal_bbox_from_ca


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2

    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0) #[40,16,16,77]
    out = out.sum(0) / out.shape[0]
    return out

def aggregate_attention_for_vis(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
            elif item.shape[1] < num_pixels:
                cur_res = int(math.sqrt(item.shape[1]))
                cross_maps = item.reshape(1, -1, cur_res, cur_res, item.shape[-1])[select]
                cross_maps = rearrange(cross_maps, 'b h w c -> b c h w')
                cross_maps = torch.nn.functional.interpolate(cross_maps, size=(res,res),mode='nearest', )
                cross_maps = rearrange(cross_maps, 'b c h w -> b h w c')
                out.append(cross_maps)
    out = torch.cat(out, dim=0) #[40,16,16,77]
    out = out.sum(0) / out.shape[0]
    return out


def aggregate_attention_intermediate(
        attention_store: AttentionStore,
        res: int,
        from_where: List[str],
        from_res: List[int],
        is_cross: bool,
        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = [r ** 2 for r in from_res]
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] in num_pixels:
                cur_res = int(math.sqrt(item.shape[1]))
                cross_maps = item.reshape(1, -1, cur_res, cur_res, item.shape[-1])[select]
                cross_maps = rearrange(cross_maps, 'b h w c -> b c h w')
                cross_maps = torch.nn.functional.interpolate(cross_maps, size=(res,res),mode='nearest', )
                cross_maps = rearrange(cross_maps, 'b c h w -> b h w c')
                out.append(cross_maps)
    out = torch.cat(out, dim=0) #[40,16,16,77]
    out = out.sum(0) / out.shape[0]
    return out