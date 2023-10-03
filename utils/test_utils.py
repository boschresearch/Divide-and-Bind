import torch
from typing import List, Dict
from PIL import Image
from IPython.display import display

from divide_and_bind.pipeline_divide_and_bind import DivideAndBindPipeline, get_indices_to_alter
from divide_and_bind.config import RunConfig
from utils.ptp_utils import AttentionStore
from utils import ptp_utils


def run_on_prompt(
        prompt: List[str],
        model: DivideAndBindPipeline,
        controller: AttentionStore,
        token_indices: List[int],
        seed: torch.Generator,
        config: RunConfig,
        save_attn_everystep=False,
        loss_mode: str = 'max',
        max_refinement_steps_init_step: int = 20,
        color_index_list=None,
        threshold_indicator='max_attn',
) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(
        prompt=prompt,
        attention_store=controller,
        indices_to_alter=token_indices,
        attention_res=config.attention_res,
        guidance_scale=config.guidance_scale,
        generator=seed,
        num_inference_steps=config.n_inference_steps,
        max_iter_to_alter=config.max_iter_to_alter,
        run_standard_sd=config.run_standard_sd,
        thresholds=config.thresholds,
        scale_factor=config.scale_factor,
        scale_range=config.scale_range,
        smooth_attentions=config.smooth_attentions,
        sigma=config.sigma,
        kernel_size=config.kernel_size,
        save_attn_everystep=save_attn_everystep,
        loss_mode=loss_mode,
        max_refinement_steps_init_step=max_refinement_steps_init_step,
        color_index_list=color_index_list,
        threshold_indicator=threshold_indicator,
    )
    image = outputs.images[0]
    return image


def run_and_display(
        model,
        prompts: List[str],
        controller: AttentionStore,
        indices_to_alter: List[int],
        generator: torch.Generator,
        run_standard_sd: bool = False,
        scale_factor: int = 20,
        thresholds: Dict[int, float] = {10: 0.5, 20: 0.8},
        max_iter_to_alter: int = 25,
        save_attn_everystep: bool = False,
        display_output: bool = False,
        loss_mode: str = 'max',
        max_refinement_steps_init_step: int = 20,
        color_index_list=None,
        threshold_indicator='max_attn',
):
    config = RunConfig(
        prompt=prompts[0],
        run_standard_sd=run_standard_sd,
        scale_factor=scale_factor,
        thresholds=thresholds,
        max_iter_to_alter=max_iter_to_alter)
    image = run_on_prompt(
        model=model,
        prompt=prompts,
        controller=controller,
        token_indices=indices_to_alter,
        seed=generator,
        save_attn_everystep=save_attn_everystep,
        config=config, loss_mode=loss_mode,
        max_refinement_steps_init_step=max_refinement_steps_init_step,
        color_index_list=color_index_list,
        threshold_indicator=threshold_indicator,
    )
    if display_output:
        display(image)
    return image


def default_param(loss_mode):
    assert loss_mode in ['max', 'tv', 'tv_bind']
    max_refinement_steps_init_step = 50
    if loss_mode == 'max':
        threshold_indicator = 'max_attn'
        max_refinement_steps_init_step  = 20
        thresholds = {0: 0.05, 10: 0.5, 20: 0.8}
    elif loss_mode == 'tv':
        threshold_indicator = 'tv_loss'
        max_refinement_steps_init_step = 50
        thresholds = {0: 0.05, 10: 0.2, 20: 0.3}
    elif loss_mode in ['tv_bind']:
        threshold_indicator = 'tv_loss'
        thresholds = {0: 0.05, 10: 0.2, 20: 0.3}  # 'tv_loss'
    elif loss_mode == 'tv_max':
        threshold_indicator = 'max_attn'
        thresholds = {0: 0.05, 10: 0.2, 20: 0.3}
    else:
        raise NotImplementedError

    parm_dict = {
        'threshold_indicator': threshold_indicator,
        'thresholds': thresholds,
        'max_refinement_steps_init_step': max_refinement_steps_init_step,
    }
    return parm_dict
