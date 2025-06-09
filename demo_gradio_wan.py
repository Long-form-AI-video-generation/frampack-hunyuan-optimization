# from diffusers_helper.hf_login import login

# import os

# os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# import gradio as gr
import torch
# import traceback
# import einops
# import safetensors.torch as sf
# from torchvision import transforms
# import numpy as np
# import argparse
# import math

# from PIL import Image
# from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
# from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
# from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
# from diffusers_helper.models.wan_video_packed import Wan21VideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
# from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
# from diffusers_helper.thread_utils import AsyncStream, async_run
# from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
# from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLWan
# from diffusers_helper.clip_vision import hf_clip_vision_encode
# from diffusers_helper.bucket_tools import find_nearest_bucket

from diffusers_helper.wan_components.wan.modules.vae import WanVAE_  # Adjust the import based on actual class names
from transformers import T5EncoderModel
from diffusers_helper.wan_components.wan.modules.t5 import T5Model , T5Encoder  # Adjust as needed
from diffusers_helper.wan_components.wan.modules.tokenizers import HuggingfaceTokenizer  # Adjust as needed
from diffusers_helper.wan_components.wan.modules.clip import XLMRobertaCLIP, CLIPModel  # Adjust as needed
# from diffusers_helper.wan_components.wan.modules.model import WanModel  

# def load_image_as_numpy(image_path):
    
#     image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
#     image_np = np.array(image)
#     return image_np


# parser = argparse.ArgumentParser()
# parser.add_argument('--share', action='store_true')
# parser.add_argument("--server", type=str, default='0.0.0.0')
# parser.add_argument("--port", type=int, required=False)
# parser.add_argument("--inbrowser", action='store_true')
# args = parser.parse_args()

# # for win desktop probably use --server 127.0.0.1 --inbrowser
# # For linux server probably use --server 127.0.0.1 or do not use any cmd flags

# print(args)

# free_mem_gb = get_cuda_free_memory_gb(gpu)
# high_vram = free_mem_gb > 60

# print(f'Free VRAM {free_mem_gb} GB')
# print(f'High-VRAM Mode: {high_vram}')
# model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
image_processor = CLIPImageProcessor.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='image_processor')
image_encoder = CLIPVisionModelWithProjection.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
vae=AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='vae', torch_dtype=torch.float16).cpu()
# transformer = Wan21VideoTransformer3DModelPacked.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",subfolder='transformer', torch_dtype=torch.bfloat16)
# transformer = transformer.to('cpu')
# Initialize models

text_encoder = T5Encoder(
    vocab=32128,       # or tokenizer.vocab_size
    dim=512,
    dim_attn=64,
    dim_ffn=2048,
    num_heads=8,
    num_layers=6,
    num_buckets=32,
    shared_pos=True,
    dropout=0.1
)


tokenizer = HuggingfaceTokenizer(
    name="t5-small",
    seq_len=64,                  
    clean="canonicalize"         
)


# text = "This is a test sentence."
# input_ids, attention_mask = tokenizer("This is a test sentence.", return_mask=True)

# encoded = text_encoder(input_ids)
# print(encoded.shape)


# image = Image.open("random-image.jpg").convert("RGB")  # or already loaded PIL.Image

# # Process the image
# processed = image_processor(images=image, return_tensors="pt")  # returns a dict

# # Access the tensor
# pixel_values = processed['pixel_values']  # shape: (1, 3, 224, 224)

# print("Pixel values shape:", pixel_values.shape)  # Should be (1, 3, 224, 224)
# print("Pixel values dtype:", pixel_values.dtype)

# with torch.no_grad():
#     outputs = image_encoder(pixel_values=processed['pixel_values'])

# # 'image_embeds' is the projected feature vector
# image_features = outputs.image_embeds  # shape (1, projection_dim=1024)
# print("Image features shape:", image_features.shape)
# print("Image features dtype:", image_features.dtype)

# print('---------------------------------------------------------------')
# dummy_video = torch.randn(1, 3, 4, 64, 64).half().cpu()  # match float16 and 5D shape

# # Encode-decode cycle
# with torch.no_grad():
#     latent = vae.encode(dummy_video).latent_dist.sample()
#     recon = vae.decode(latent).sample

# print("Input shape:", dummy_video.shape)
# print("Encoded shape:", latent.shape)
# print("Decoded shape:", recon.shape)

# demo_gradio_wan21.py
# Modified version of demo_gradio.py for WAN 2.1

import os
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
import gradio as gr
import torch
import traceback
import einops
import numpy as np
import argparse

from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from diffusers_helper.utils import save_bcthw_as_mp4, generate_timestamp
# from diffusers_helper.models.wan_video_packed import (
#     Wan21VideoTransformer3DModelPacked, 
#     Wan21VAEEncoder, 
#     Wan21VAEDecoder,
#     create_wan21_1_3b_config,
#     create_wan21_14b_config
# )
# from diffusers_helper.pipelines.k_diffusion_wan21 import sample_wan21
# from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb
# from diffusers_helper.thread_utils import AsyncStream, async_run
# from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
def get_actual_free_memory():
    """Get actual free GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)
        
        # Actual free memory
        free_memory = total_memory - reserved_memory
        
        print(f"💾 GPU Memory Status:")
        print(f"   Total: {total_memory / (1024**3):.1f} GB")
        print(f"   Allocated: {allocated_memory / (1024**3):.1f} GB")
        print(f"   Reserved: {reserved_memory / (1024**3):.1f} GB")
        print(f"   Actually Free: {free_memory / (1024**3):.1f} GB")
        
        return free_memory / (1024**3)
    return 0
if torch.cuda.is_available():
        print("🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check if there are any models in GPU memory
        allocated = torch.cuda.memory_allocated(0)
        if allocated > 100 * 1024 * 1024:  # More than 100MB
            print(f"⚠️ Warning: {allocated / (1024**3):.1f} GB still allocated on GPU")
            print("   You may need to restart Python to fully clear memory")
from diffusers_helper.models.wan_video_packed import WANDiTBlock, test_wan_dit_block
from diffusers_helper.models.random import ModifiedWANDiTBlock

model =  ModifiedWANDiTBlock.from_wan_pretrained(
            wan_model_path='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers' ,subfolder='transformer',  # Replace with actual WAN model path
            framepack_model_path='lllyasviel/FramePackI2V_HY'
        )

# from_hunyuan_pretrained(
#         'lllyasviel/FramePackI2V_HY', 
#         torch_dtype=torch.bfloat16
#     )
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
if torch.cuda.is_available():
        device = 'cpu'
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_gb = get_actual_free_memory()
        print(f"💾 Available GPU memory: {free_gb:.1f} GB")
        
        if free_gb > 8.0:  # Need at least 8GB for full model
            print("   Moving to GPU...")
            model = model.cpu()
            batch_size = 1
            hidden_states = torch.randn(batch_size, 16, 4, 32, 32, device=device)
            timestep = torch.tensor([500], device=device)
            encoder_hidden_states = torch.randn(batch_size, 77, 4096, device=device)
            encoder_attention_mask = torch.ones(batch_size, 77, device=device).bool()
            pooled_projections = torch.randn(batch_size, 768, device=device)
            guidance = torch.tensor([7.5], device=device)
            
            print("✅ Model on GPU")
        else:
            print("   ⚠️ Low GPU memory, keeping on CPU")
            print("   For Day 1 testing, CPU inference is fine!")
else:
        print("   No CUDA available, using CPU")
    
    # Create test inputs


# print(f"\n🧪 Testing forward pass... here ")
prompt = "A fantasy landscape with mountains and rivers"

input_ids, attention_mask = tokenizer(prompt,return_mask=True)

with torch.no_grad():
    
    
#     # Get T5 embeddings
#     encoder_outputs = text_encoder(input_ids=text_input_ids)
    prompt_embeds = text_encoder(input_ids)
    prompt_mask =attention_mask
    
    expected_pooled_dim = 768  # CLIP text encoder output size
    batch_size = prompt_embeds.shape[0]
    # For simplicity, use pooled_projections as None or zeros for now
    pooled_projections = torch.zeros(batch_size, expected_pooled_dim, 
                                    device=prompt_embeds.device, 
                                    dtype=prompt_embeds.dtype)

    # Prepare negative prompt (empty or simple)
    negative_prompt = ""
    neg_inputs, attention_mask_nega = tokenizer(negative_prompt, return_mask=True)
    negative_prompt_embeds = text_encoder(neg_inputs)
    negative_prompt_mask = attention_mask_nega
    negative_pooled_projections =torch.zeros(batch_size, expected_pooled_dim,
                                            device=negative_prompt_embeds.device,
                                            dtype=negative_prompt_embeds.dtype)

    expected_text_dim = 4096  # Original HunyuanVideo text encoder size
    current_text_dim = prompt_embeds.shape[-1]  # 512 from T5
    
    if current_text_dim != expected_text_dim:
        print(f"🔧 Projecting text embeddings: {current_text_dim} → {expected_text_dim}")
        
        # Create projection layer
        text_projection = torch.nn.Linear(current_text_dim, expected_text_dim).to(prompt_embeds.device)
        torch.nn.init.normal_(text_projection.weight, std=0.02)
        torch.nn.init.zeros_(text_projection.bias)
        
        # Project positive embeddings
        batch_size, seq_len = prompt_embeds.shape[:2]
        prompt_embeds_flat = prompt_embeds.view(-1, current_text_dim)
        projected_embeds = text_projection(prompt_embeds_flat)
        prompt_embeds = projected_embeds.view(batch_size, seq_len, expected_text_dim)
        
        # Project negative embeddings
        batch_size, seq_len = negative_prompt_embeds.shape[:2]
        neg_embeds_flat = negative_prompt_embeds.view(-1, current_text_dim)
        projected_neg_embeds = text_projection(neg_embeds_flat)
        negative_prompt_embeds = projected_neg_embeds.view(batch_size, seq_len, expected_text_dim)
        
        print(f"✅ Projected embeddings shape: {prompt_embeds.shape}")

    # 4. Run sample_hunyuan
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    print(f"🔧 Moving tensors to device: {model_device}, dtype: {model_dtype}")

    # Move text embeddings to model device and dtype
    prompt_embeds = prompt_embeds.to(device=model_device, dtype=model_dtype)
    prompt_mask = prompt_mask.to(device=model_device)
    pooled_projections = pooled_projections.to(device=model_device, dtype=model_dtype)

    negative_prompt_embeds = negative_prompt_embeds.to(device=model_device, dtype=model_dtype)
    negative_prompt_mask = negative_prompt_mask.to(device=model_device)
    negative_pooled_projections = negative_pooled_projections.to(device=model_device, dtype=model_dtype)

    print(f"✅ All tensors moved to {model_device}")
    print(f"   prompt_embeds: {prompt_embeds.shape} on {prompt_embeds.device}")
    print(f"   pooled_projections: {pooled_projections.shape} on {pooled_projections.device}")

    seed = 42  # or any seed you want
    generator = torch.Generator("cpu").manual_seed(seed)
    
    # 4. Run sample_hunyuan with generator
    outputs = sample_hunyuan(
        transformer=model,
        sampler='unipc',
        width=64,  # small size for quick test
        height=64,
        frames=4,
        num_inference_steps=10,
        batch_size=1,
        generator=generator,  # CPU generator is fine
        prompt_embeds=prompt_embeds,  # Now on correct device
        prompt_embeds_mask=prompt_mask,  # Now on correct device
        prompt_poolers=pooled_projections,  # Now on correct device
        negative_prompt_embeds=negative_prompt_embeds,  # Now on correct device
        negative_prompt_embeds_mask=negative_prompt_mask,  # Now on correct device
        negative_prompt_poolers=negative_pooled_projections,  # Now on correct device
        device=model_device,  # Use model device
        dtype=model_dtype 
    )

print("Sampled latent shape:", outputs.shape)


# with torch.no_grad():
#     output = model(
#     hidden_states=hidden_states,
#     timestep=timestep,
#     encoder_hidden_states=encoder_hidden_states,
#     encoder_attention_mask=encoder_attention_mask,
#     pooled_projections=pooled_projections,
#     guidance=guidance,
#     )

# print(f"🎉 SUCCESS!")
# print(f"Input: {hidden_states.shape}")
# print(f"Output: {output.sample.shape}")


# test_wan_dit_block()

# parser = argparse.ArgumentParser()
# parser.add_argument('--share', action='store_true')
# parser.add_argument("--server", type=str, default='0.0.0.0')
# parser.add_argument("--port", type=int, required=False)
# parser.add_argument("--inbrowser", action='store_true')
# parser.add_argument("--model_size", type=str, default="1.3b", choices=["1.3b", "14b"])
# args = parser.parse_args()

# print(args)

# free_mem_gb = get_cuda_free_memory_gb(gpu)
# high_vram = free_mem_gb > 24  # WAN 2.1 14B needs more memory

# print(f'Free VRAM {free_mem_gb} GB')
# print(f'High-VRAM Mode: {high_vram}')
# print(f'Loading WAN 2.1 {args.model_size.upper()} model')

# # Load T5 text encoder
# if args.model_size == "1.3b":
#     t5_model_name = "google/t5-v1_1-base"
#     model_config = create_wan21_1_3b_config()
#     model_path = "Wan-AI/Wan2.1-T2V-1.3B"  # HuggingFace path
# else:
#     t5_model_name = "google/t5-v1_1-xl"
#     model_config = create_wan21_14b_config()
#     model_path = "Wan-AI/Wan2.1-T2V-14B"

# text_encoder = T5EncoderModel.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='text_encoder').cpu()
# tokenizer = T5Tokenizer.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='tokenizer').cpu()

# # Initialize VAE (simplified version - in practice would load Wan-VAE)
# vae_encoder = Wan21VAEEncoder().cpu()
# vae_decoder = Wan21VAEDecoder().cpu()

# # Initialize transformer
# transformer = Wan21VideoTransformer3DModelPacked(**model_config, torch_dtype=torch.bfloat16).cpu()

# # In practice, you would load the pretrained weights here
# # transformer.load_state_dict(torch.load(model_path))

# text_encoder.eval()
# vae_encoder.eval()
# vae_decoder.eval()
# transformer.eval()

# if not high_vram:
#     # Enable memory efficient features
#     transformer.gradient_checkpointing = True

# # Set precision
# transformer.to(dtype=torch.bfloat16)
# vae_encoder.to(dtype=torch.float16)
# vae_decoder.to(dtype=torch.float16)
# text_encoder.to(dtype=torch.float16)

# # Freeze all models
# for model in [vae_encoder, vae_decoder, text_encoder, transformer]:
#     model.requires_grad_(False)

# if high_vram:
#     text_encoder.to(gpu)
#     vae_encoder.to(gpu)
#     vae_decoder.to(gpu)
#     transformer.to(gpu)

# stream = AsyncStream()
# outputs_folder = './outputs/'
# os.makedirs(outputs_folder, exist_ok=True)


# @torch.no_grad()
# def encode_prompt_wan21(prompt, text_encoder, tokenizer, max_length=77):
#     """Encode text prompt using T5"""
#     # Tokenize
#     text_inputs = tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
    
#     text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    
#     # Get T5 embeddings
#     encoder_outputs = text_encoder(input_ids=text_input_ids)
#     encoder_hidden_states = encoder_outputs.last_hidden_state
    
#     return encoder_hidden_states


# @torch.no_grad()
# def vae_encode_wan21(video, vae_encoder):
#     """Encode video using Wan-VAE encoder"""
#     # Simple encoding - in practice Wan-VAE has more complex causal structure
#     latents = vae_encoder(video)
#     return latents


# @torch.no_grad()
# def vae_decode_wan21(latents, vae_decoder):
#     """Decode latents using Wan-VAE decoder"""
#     video = vae_decoder(latents)
#     return video


# @torch.no_grad()
# def worker(prompt, n_prompt, seed, video_length, fps, steps, cfg, height, width, gpu_memory_preservation):
#     """Worker function for WAN 2.1 video generation"""
    
#     job_id = generate_timestamp()
#     total_frames = int(video_length * fps)
    
#     stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    
#     try:
#         # Text encoding
#         stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        
#         prompt_embeds = encode_prompt_wan21(prompt, text_encoder, tokenizer)
        
#         if cfg == 1:
#             negative_prompt_embeds = torch.zeros_like(prompt_embeds)
#         else:
#             negative_prompt_embeds = encode_prompt_wan21(n_prompt, text_encoder, tokenizer)
        
#         # Initialize latents
#         stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Initializing ...'))))
        
#         rnd = torch.Generator("cpu").manual_seed(seed)
        
#         # Calculate latent dimensions (simplified)
#         latent_height = height // 8
#         latent_width = width // 8
#         latent_frames = total_frames // 4  # Assuming 4x temporal compression
        
#         latents = torch.randn(
#             (1, 16, latent_frames, latent_height, latent_width), 
#             generator=rnd, 
#             device=gpu
#         ).to(dtype=torch.float32)
        
#         # Sampling
#         stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        
#         def callback(d):
#             if stream.input_queue.top() == 'end':
#                 stream.output_queue.push(('end', None))
#                 raise KeyboardInterrupt('User ends the task.')
            
#             current_step = d['i'] + 1
#             percentage = int(100.0 * current_step / steps)
#             hint = f'Sampling {current_step}/{steps}'
#             desc = f'Generating {total_frames} frames at {height}x{width}'
#             stream.output_queue.push(('progress', (None, desc, make_progress_bar_html(percentage, hint))))
#             return
        
#         # Sample using WAN 2.1 pipeline
#         generated_latents = sample_wan21(
#             transformer=transformer,
#             latents=latents,
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#             num_inference_steps=steps,
#             guidance_scale=cfg,
#             generator=rnd,
#             callback=callback,
#             device=gpu,
#             dtype=torch.bfloat16,
#         )
        
#         # Decode
#         stream.output_queue.push(('progress', (None, '', make_progress_bar_html(90, 'Decoding ...'))))
        
#         if not high_vram:
#             transformer.to(cpu)
#             vae_decoder.to(gpu)
        
#         video = vae_decode_wan21(generated_latents, vae_decoder)
        
#         # Convert to proper format and save
#         video = video.cpu()
#         output_filename = os.path.join(outputs_folder, f'{job_id}.mp4')
#         save_bcthw_as_mp4(video, output_filename, fps=fps, crf=23)
        
#         print(f'Generated video saved to {output_filename}')
#         stream.output_queue.push(('file', output_filename))
        
#     except:
#         traceback.print_exc()
    
#     stream.output_queue.push(('end', None))
#     return


# def process(prompt, n_prompt, seed, video_length, fps, steps, cfg, height, width, gpu_memory_preservation):
#     global stream
    
#     yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    
#     stream = AsyncStream()
    
#     async_run(worker, prompt, n_prompt, seed, video_length, fps, steps, cfg, height, width, gpu_memory_preservation)
    
#     output_filename = None
    
#     while True:
#         flag, data = stream.output_queue.next()
        
#         if flag == 'file':
#             output_filename = data
#             yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
        
#         if flag == 'progress':
#             preview, desc, html = data
#             yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        
#         if flag == 'end':
#             yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
#             break


# def end_process():
#     stream.input_queue.push('end')


# # Quick prompts for WAN 2.1
# quick_prompts = [
#     'A panda performs difficult skateboarding tricks on city streets',
#     'Close-up shot, ice cubes fall from a height into a glass',
#     'A snowmobiler speeding and kicking up snow on a snowy landscape',
#     'A close-up cinematic shot capturing the face of a transforming spy',
# ]
# quick_prompts = [[x] for x in quick_prompts]

# css = make_progress_bar_css()
# block = gr.Blocks(css=css).queue()

# with block:
#     gr.Markdown(f'# WAN 2.1 Video Generation ({args.model_size.upper()} Model)')
    
#     with gr.Row():
#         with gr.Column():
#             prompt = gr.Textbox(label="Prompt", value='')
#             example_quick_prompts = gr.Dataset(
#                 samples=quick_prompts, 
#                 label='Quick Prompts', 
#                 samples_per_page=1000, 
#                 components=[prompt]
#             )
#             example_quick_prompts.click(
#                 lambda x: x[0], 
#                 inputs=[example_quick_prompts], 
#                 outputs=prompt, 
#                 show_progress=False, 
#                 queue=False
#             )
            
#             with gr.Row():
#                 start_button = gr.Button(value="Start Generation")
#                 end_button = gr.Button(value="End Generation", interactive=False)
            
#             with gr.Group():
#                 n_prompt = gr.Textbox(label="Negative Prompt", value="low quality, blurry")
#                 seed = gr.Number(label="Seed", value=42, precision=0)
                
#                 video_length = gr.Slider(
#                     label="Video Length (seconds)", 
#                     minimum=1, 
#                     maximum=10, 
#                     value=5, 
#                     step=0.5
#                 )
#                 fps = gr.Slider(label="FPS", minimum=8, maximum=30, value=24, step=1)
                
#                 with gr.Row():
#                     height = gr.Slider(
#                         label="Height", 
#                         minimum=256, 
#                         maximum=720, 
#                         value=480, 
#                         step=16
#                     )
#                     width = gr.Slider(
#                         label="Width", 
#                         minimum=256, 
#                         maximum=1280, 
#                         value=640, 
#                         step=16
#                     )
                
#                 steps = gr.Slider(
#                     label="Steps", 
#                     minimum=10, 
#                     maximum=50, 
#                     value=50 if args.model_size == "14b" else 40, 
#                     step=1
#                 )
#                 cfg = gr.Slider(
#                     label="CFG Scale", 
#                     minimum=1.0, 
#                     maximum=15.0, 
#                     value=7.5, 
#                     step=0.1
#                 )
                
#                 gpu_memory_preservation = gr.Slider(
#                     label="GPU Preserved Memory (GB)", 
#                     minimum=2, 
#                     maximum=16, 
#                     value=4, 
#                     step=0.5,
#                     info="Increase if you encounter OOM errors"
#                 )
        
#         with gr.Column():
#             preview_image = gr.Image(label="Preview", height=200, visible=False)
#             result_video = gr.Video(
#                 label="Generated Video", 
#                 autoplay=True, 
#                 show_share_button=False, 
#                 height=512, 
#                 loop=True
#             )
#             progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
#             progress_bar = gr.HTML('', elem_classes='no-generating-animation')
    
#     gr.Markdown(
#         f'Using WAN 2.1 {args.model_size.upper()} model. '
#         f'The 1.3B model requires ~8GB VRAM, while 14B requires ~24GB VRAM.'
#     )
    
#     ips = [prompt, n_prompt, seed, video_length, fps, steps, cfg, height, width, gpu_memory_preservation]
#     start_button.click(
#         fn=process, 
#         inputs=ips, 
#         outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
#     )
#     end_button.click(fn=end_process)

# block.launch(
#     server_name=args.server,
#     server_port=args.port,
#     share=args.share,
#     inbrowser=args.inbrowser,
# )

