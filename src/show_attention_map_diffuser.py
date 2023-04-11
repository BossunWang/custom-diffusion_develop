# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from src import diffuser_training 
from PIL import Image


class AttentionControl(abc.ABC):
    def __init__(self, low_resource):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.low_resource = low_resource

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    def __init__(self, low_resource):
        super(AttentionStore, self).__init__(low_resource)
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def sample(ckpt, delta_ckpt, from_file, prompt, compress, batch_size, freeze_model):
    model_id = ckpt
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    LOW_RESOURCE = True
    controller = AttentionStore(LOW_RESOURCE)

    outdir = 'outputs/txt2img-samples'
    os.makedirs(outdir, exist_ok=True)
    if delta_ckpt is not None:
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, compress, freeze_model)
        outdir = os.path.dirname(delta_ckpt)

    all_images = []
    # if prompt is not None:
    #     images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., eta=1.).images
    #     all_images += images
    #     images = np.hstack([np.array(x) for x in images])
    #     images = Image.fromarray(images)
    #     # takes only first 50 characters of prompt to name the image file
    #     name = '-'.join(prompt[:50].split())
    #     images.save(f'{outdir}/{name}.png')
    # else:
    #     print(f"reading prompts from {from_file}")
    #     with open(from_file, "r") as f:
    #         data = f.read().splitlines()
    #         data = [[prompt]*batch_size for prompt in data]
    #
    #     for prompt in data:
    #         images = pipe(prompt, num_inference_steps=200, guidance_scale=6., eta=1.).images
    #         all_images += images
    #         images = np.hstack([np.array(x) for x in images], 0)
    #         images = Image.fromarray(images)
    #         # takes only first 50 characters of prompt to name the image file
    #         name = '-'.join(prompt[0][:50].split())
    #         images.save(f'{outdir}/{name}.png')
    #
    # os.makedirs(f'{outdir}/samples', exist_ok=True)
    # for i, im in enumerate(all_images):
    #     im.save(f'{outdir}/samples/{i}.jpg')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query', default=None,
                        type=str)
    parser.add_argument('--from-file', help='path to prompt file', default='./',
                        type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompt, args.compress, args.batch_size, args.freeze_model)
