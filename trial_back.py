
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from typing import Optional, List, Tuple, Dict

import safetensors.torch as sf
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image

from diffusers_helper.wan_components.wan.distributed.fsdp import shard_model
from diffusers_helper.wan_components.wan.modules.clip import CLIPModel
from diffusers_helper.wan_components.wan.modules.model import WanModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection,UMT5EncoderModel,AutoTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKLWan
from diffusers_helper.wan_components.wan.modules.vae import WanVAE
from diffusers_helper.wan_components.wan.modules.t5 import T5Encoder
from diffusers_helper.wan_components.wan.modules.tokenizers import HuggingfaceTokenizer
from diffusers_helper.wan_components.wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from diffusers_helper.wan_components.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanFramePackConfig:
    """Configuration class for WAN FramePack integration"""
    
    def __init__(self):
        # Model parameters
        self.num_train_timesteps = 1000
        self.param_dtype = torch.bfloat16
        self.vae_stride = [4, 8, 8]
        self.patch_size = [1, 2, 2]
        
        # FramePack parameters
        self.latent_window_size = 9
        self.max_video_length = 120.0  # seconds
        self.default_fps = 30
        self.use_teacache = True
        
        # Sampling parameters
        self.default_sampling_steps = 25
        self.default_guidance_scale = 10.0
        self.default_shift = 5.0
        self.sample_neg_prompt = "low quality, worst quality, blurry, distorted"
        
        # Memory management
        self.offload_models = True
        self.quantization_type = 'fp8'  # 'fp8', 'fp16', 'int8'
        self.enable_gradient_checkpointing = True
        
        # Output settings
        self.output_fps = 30
        self.output_crf = 16  # Video compression quality
        
        # Multi-GPU settings
        self.use_multi_gpu = True
        self.model_parallel_size = 4  # Number of GPUs to use for model parallel


class FramePackSampler:
    """Implements FramePack's progressive sampling technique for long video generation"""
    
    def __init__(self, latent_window_size: int = 9, vae_stride: List[int] = [4, 8, 8]):
        self.latent_window_size = latent_window_size
        self.vae_stride = vae_stride
        
    def calculate_sections(self, total_second_length: float, fps: int = 30) -> Tuple[int, List[int]]:
        """Calculate the number of sections and padding pattern for generation"""
        total_frames = int(total_second_length * fps)
        frames_per_section = self.latent_window_size * 4
        total_latent_sections = (total_frames) / frames_per_section
        total_latent_sections = int(max(round(total_latent_sections), 1))
        
        # FramePack's padding pattern
        if total_latent_sections > 4:
            # Duplicate some items for better quality when sections > 4
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            latent_paddings = list(reversed(range(total_latent_sections)))
            
        return total_latent_sections, latent_paddings
    
    def prepare_indices(self, latent_padding: int, include_4x: bool = True) -> Dict[str, torch.Tensor]:
        """Prepare indices for clean latents based on padding"""
        latent_padding_size = latent_padding * self.latent_window_size
        
        if include_4x:
            indices = torch.arange(0, sum([1, latent_padding_size, self.latent_window_size, 1, 2, 16])).unsqueeze(0)
            splits = indices.split([1, latent_padding_size, self.latent_window_size, 1, 2, 16], dim=1)
            
            return {
                'clean_latent_indices_pre': splits[0],
                'blank_indices': splits[1],
                'latent_indices': splits[2],
                'clean_latent_indices_post': splits[3],
                'clean_latent_2x_indices': splits[4],
                'clean_latent_4x_indices': splits[5],
                'clean_latent_indices': torch.cat([splits[0], splits[3]], dim=1)
            }
        else:
            indices = torch.arange(0, sum([1, latent_padding_size, self.latent_window_size, 1])).unsqueeze(0)
            splits = indices.split([1, latent_padding_size, self.latent_window_size, 1], dim=1)
            
            return {
                'clean_latent_indices_pre': splits[0],
                'blank_indices': splits[1],
                'latent_indices': splits[2],
                'clean_latent_indices_post': splits[3],
                'clean_latent_indices': torch.cat([splits[0], splits[3]], dim=1)
            }

    def soft_append_pixels(self, history: torch.Tensor, current: torch.Tensor, 
                          overlap: int = 0) -> torch.Tensor:
        """Soft blending of overlapping frames (adapted from FramePack)"""
        if overlap <= 0:
            return torch.cat([history, current], dim=2)

        assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
        
        weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device)
        weights = weights.view(1, 1, -1, 1, 1)
        
        blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)
        
        return output.to(history)


class ModelParallelWan(torch.nn.Module):
    """Wrapper for model parallel WAN across multiple GPUs"""
    
    def __init__(self, model_path: str, num_gpus: int = 3):
        super().__init__()
        # self.num_gpus = num_gpus
        self.num_gpus =3
        self.devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        
        # Load the model architecture
        print("Creating WAN model architecture...")
        self.model = WanModel(
            model_type='i2v',
            num_layers=32,
            dim=5120,
            ffn_dim=13824,
            in_dim=36
        )
        print("DEBUG: Model attributes:")
        for name, module in self.model.named_children():
            print(f"  - {name}: {type(module).__name__}")
        # Load weights and distribute across GPUs
        self._load_and_distribute_weights(model_path)
        
    def _load_and_distribute_weights(self, model_path: str):
        """Load weights and distribute layers across GPUs"""
        print(f"Loading weights from {model_path} and distributing across {self.num_gpus} GPUs...")
        
        # Load state dict
        state_dict = sf.load_file(model_path, device='cpu')
        
        # Debug: Check what embedder keys are available
        embedder_keys = [k for k in state_dict.keys() if 'embedder' in k.lower()]
        print(f"DEBUG: Found embedder keys: {embedder_keys[:5]}...")  # Show first 5
        
        # Calculate layer distribution
        num_layers = len(self.model.blocks)
        layers_per_gpu = num_layers // self.num_gpus
        
        # First, ensure x_embedder is properly initialized
        if hasattr(self.model, 'x_embedder'):
            print("DEBUG: Model has x_embedder")
            # Move x_embedder to first GPU
            self.model.x_embedder = self.model.x_embedder.to(self.devices[0])
            
            # Load x_embedder weights
            x_embedder_keys = [k for k in state_dict.keys() if 'x_embedder' in k]
            print(f"DEBUG: Found {len(x_embedder_keys)} x_embedder keys")
            
            for key in x_embedder_keys:
                if key in state_dict:
                    # Remove 'x_embedder.' prefix to get the parameter name
                    param_name = key.replace('x_embedder.', '')
                    try:
                        # Access the parameter in the x_embedder module
                        module = self.model.x_embedder
                        for part in param_name.split('.')[:-1]:
                            module = getattr(module, part)
                        setattr(module, param_name.split('.')[-1], 
                            torch.nn.Parameter(state_dict[key].to(self.devices[0])))
                    except:
                        # Alternative approach
                        self.model.x_embedder.load_state_dict(
                            {k.replace('x_embedder.', ''): v for k, v in state_dict.items() 
                            if k.startswith('x_embedder.')},
                            strict=False
                        )
                        break
        
        # Distribute embedding layers to first GPU
        embed_keys = [k for k in state_dict.keys() if any(x in k for x in ['patch_embed', 'pos_embed'])]
        for key in embed_keys:
            if key in state_dict:
                param = state_dict[key].to(self.devices[0])
                self.model.state_dict()[key].data.copy_(param)
        
        # Move other embedding layers to first GPU
        if hasattr(self.model, 'patch_embed'):
            self.model.patch_embed = self.model.patch_embed.to(self.devices[0])
            
        # Distribute transformer blocks across GPUs
        for i, block in enumerate(self.model.blocks):
            gpu_idx = min(i // layers_per_gpu, self.num_gpus - 1)
            device = self.devices[gpu_idx]
            
            # Move block to appropriate GPU
            block = block.to(device)
            
            # Load weights for this block
            block_keys = [k for k in state_dict.keys() if f'blocks.{i}.' in k]
            block_state = {k.replace(f'blocks.{i}.', ''): state_dict[k].to(device) 
                        for k in block_keys}
            block.load_state_dict(block_state, strict=False)
                        
        # Move output layers to last GPU
        output_keys = [k for k in state_dict.keys() if any(x in k for x in ['final_layer', 'norm_out', 'linear_out'])]
        for key in output_keys:
            if key in state_dict and 'blocks.' not in key:
                param = state_dict[key].to(self.devices[-1])
                self.model.state_dict()[key].data.copy_(param)
                
        if hasattr(self.model, 'final_layer'):
            self.model.final_layer = self.model.final_layer.to(self.devices[-1])
            
        del state_dict
        torch.cuda.empty_cache()
        
        print("Model distributed across GPUs successfully")
        
    def forward(self, x, t, context, seq_len, clip_fea, y=None):
        """Forward pass with model parallelism"""
        # Start on first GPU
        current_device = self.devices[0]
        
        # Calculate context_lens based on the context tensor
        if context is not None:
            context_lens = torch.tensor([context.shape[1]], dtype=torch.long, device=context.device)
        else:
            context_lens = None
        
        # Debug: Check input format
        print(f"DEBUG: Input x type: {type(x)}, length: {len(x) if isinstance(x, list) else 'N/A'}")
        if isinstance(x, list) and len(x) > 0:
            print(f"DEBUG: First x element shape: {x[0].shape}")
        
        # Initial embeddings on first GPU - this is CRITICAL
        if hasattr(self.model, 'x_embedder'):
            # Move x to device and ensure float32
            if isinstance(x, list):
                x_input = [xi.to(current_device).to(torch.float32) for xi in x]
            else:
                x_input = [x.to(current_device).to(torch.float32)]
            
            print(f"DEBUG: Before x_embedder, x_input[0] shape: {x_input[0].shape}")
            
            # Call x_embedder - this should transform [20, 9, 92, 68] to [*, 5120]
            x = self.model.x_embedder(x_input)
            
            # Debug: Check x_embedder output
            print(f"DEBUG: After x_embedder, x type: {type(x)}")
            if isinstance(x, list):
                print(f"DEBUG: x is list with length: {len(x)}")
                if len(x) > 0 and isinstance(x[0], torch.Tensor):
                    print(f"DEBUG: First element shape: {x[0].shape}")
            elif isinstance(x, torch.Tensor):
                print(f"DEBUG: x tensor shape: {x.shape}")
        else:
            print("ERROR: Model doesn't have x_embedder!")
            raise AttributeError("Model missing x_embedder")
        
        # The model might expect x to be a single tensor after embedding
        # Try to extract it if it's a single-element list
        if isinstance(x, list) and len(x) == 1 and isinstance(x[0], torch.Tensor):
            x = x[0]
        
        # Now x should be a tensor with proper shape
        if isinstance(x, torch.Tensor):
            print(f"DEBUG: Final x shape before blocks: {x.shape}")
        else:
            print(f"ERROR: x is not a tensor, it's {type(x)}")
        
        # Ensure all inputs are float32
        if isinstance(x, torch.Tensor):
            x = x.to(torch.float32)
        else:
            raise ValueError(f"Expected x to be a tensor after embedding, but got {type(x)}")
        
        t = t.to(torch.float32) if isinstance(t, torch.Tensor) else t
        context = context.to(torch.float32) if context is not None else context
        clip_fea = clip_fea.to(torch.float32) if clip_fea is not None else clip_fea
        
        if y is not None and isinstance(y, list):
            y = [yi.to(torch.float32) for yi in y]
                
        # Pass through transformer blocks
        for i, block in enumerate(self.model.blocks):
            gpu_idx = min(i // (len(self.model.blocks) // self.num_gpus), self.num_gpus - 1)
            next_device = self.devices[gpu_idx]
            
            # Move tensors to next device if needed
            if next_device != current_device:
                x = x.to(next_device)
                t = t.to(next_device)
                if context is not None:
                    context = context.to(next_device)
                if clip_fea is not None:
                    clip_fea = clip_fea.to(next_device)
                if context_lens is not None:
                    context_lens = context_lens.to(next_device)
                if y is not None:
                    y = [yi.to(next_device) for yi in y]
                current_device = next_device
            
            # Debug first block
            if i == 0:
                print(f"DEBUG: Before first block, x shape: {x.shape}")
                
            # Apply block
            x = block(x, t, context, seq_len, clip_fea, y, context_lens)
            
        # Final layer on last GPU
        if hasattr(self.model, 'final_layer'):
            x = x.to(self.devices[-1])
            x = self.model.final_layer(x)
        
        # Return as list if that's what's expected
        return [x]

class WanI2VFramePack:
    """WAN Image-to-Video model with FramePack sampling integration and multi-GPU support"""
    
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        use_quantization: Optional[str] = None,
        quantized_model_path: Optional[str] = None,
        use_multi_gpu: bool = True,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.device2= torch.device(f"cuda:1")
        self.device3= torch.device(f"cuda:2")
        self.device4= torch.device(f"cuda:3")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.use_quantization = use_quantization
        self.quantized_model_path = quantized_model_path
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # Initialize components
        shard_fn = partial(shard_model, device_id=device_id)
        
        # Use smaller text encoder for memory efficiency
        self.text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl").cpu()
        self.tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
        
        # VAE
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", 
            subfolder='vae'
        )

        # CLIP
        self.clip_processor = CLIPImageProcessor.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='image_processor')
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        # Load main model with multi-GPU support
        if self.use_multi_gpu:
            print(f"Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
            self.model = ModelParallelWan(checkpoint_dir, num_gpus=torch.cuda.device_count())
        else:
            # Single GPU fallback
            print("Using single GPU setup")
            self.model = self._load_single_gpu_model(checkpoint_dir)

        self.sample_neg_prompt = config.sample_neg_prompt
        
        # Initialize FramePack sampler
        self.framepack_sampler = FramePackSampler(
            latent_window_size=9,
            vae_stride=self.vae_stride
        )
        
        
    


    def _load_single_gpu_model(self, model_path: str):
        """Load model for single GPU with FP16"""
        print(f"Loading FP16 model from {model_path}")
        
        # Clear GPU memory first
        torch.cuda.empty_cache()
        
        # Load state dict to CPU first
        print("Loading state dict to CPU...")
        state_dict = sf.load_file(model_path, device='cpu')

        # Create model on CPU
        print("Creating model on CPU...")
        model = WanModel(
            model_type='i2v',
            num_layers=32,
            dim=5120,
            ffn_dim=13824,
            in_dim=36 
        )
    
        # Load weights on CPU
        print("Loading weights...")
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        torch.cuda.empty_cache()
        
        # Convert to FP16 while still on CPU
        print("Converting to FP16...")
        model = model.to(dtype=torch.float16)
        
        print("Model loaded successfully on CPU")
        
        return model.eval().requires_grad_(False)

    def clear_gpu_memory(self):
        """Aggressively clear GPU memory across all devices"""
        # Move all models to CPU
        if hasattr(self, 'text_encoder'):
            self.text_encoder.cpu()
        if hasattr(self, 'clip_model'):
            self.clip_model.cpu()
        if hasattr(self, 'vae'):
            self.vae.cpu()
        if hasattr(self, 'model') and not self.use_multi_gpu:
            self.model.cpu()
        
        # Clear GPU cache on all devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()

    def prepare_wan_noise_inputs(self, noise_latents: torch.Tensor, 
                                clean_latents: torch.Tensor = None) -> List[torch.Tensor]:
        if noise_latents.dim() == 5:  # (B, T, C, H, W) or (B, C, T, H, W)
            if noise_latents.shape[1] < 5:  # Probably (B, T, C, H, W)
                noise_latents = noise_latents.permute(0, 2, 1, 3, 4)  # → (B, C, T, H, W)

            # Split batch
            noise_list = [noise_latents[i] for i in range(noise_latents.shape[0])]
        else:
            raise ValueError("Expected 5D input tensor [B, T, C, H, W] or [B, C, T, H, W]")

        return noise_list

    def prepare_wan_conditional_inputs(
        self, 
        start_latent: torch.Tensor, 
        reference_noise: List[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Create conditioning tensors that match reference_noise exactly"""
        import torch.nn.functional as F
        if reference_noise is None:
            raise ValueError("reference_noise is required to determine target shape")
        
        # Extract the conditioning frame
        if start_latent.dim() == 5:  # (B, C, T, H, W)
            if start_latent.shape[2] == 1:
                conditioning_frame = start_latent.squeeze(2)  # (B, C, H, W)
            else:
                conditioning_frame = start_latent[:, :, 0:1, :, :].squeeze(2)  # Take first frame
        elif start_latent.dim() == 4:  # (B, C, H, W)
            conditioning_frame = start_latent
        elif start_latent.dim() == 3:  # (C, H, W)
            conditioning_frame = start_latent.unsqueeze(0)  # (1, C, H, W)
        else:
            raise ValueError(f"Unexpected start_latent shape: {start_latent.shape}")
        
        y_list = []
        for i, ref_tensor in enumerate(reference_noise):
            # ref_tensor shape: [C, T, H, W]
            C, T, H, W = ref_tensor.shape
            
            # Extract single conditioning frame
            if conditioning_frame.shape[0] > 1:
                single_frame = conditioning_frame[i]  # (C, H, W)
            else:
                single_frame = conditioning_frame[0]  # (C, H, W)
            
            # Resize conditioning frame to match reference spatial dimensions
            if single_frame.shape[-2:] != (H, W):
                single_frame = F.interpolate(
                    single_frame.unsqueeze(0),  # (1, C, H, W)
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # (C, H, W)
            
            # Create conditioning tensor - repeat first frame
            y_tensor = single_frame.unsqueeze(1).repeat(1, T, 1, 1)  # (C, T, H, W)
            
            y_list.append(y_tensor)
        
        return y_list
    
    def log_gpu_memory(self, device):
        """Logs total, allocated, reserved, and free memory for a given CUDA device."""
        # device = torch.device(device)

        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # MB
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # MB
        free_inside_reserved = reserved - allocated

        device_index = device.index if device.index is not None else 0
        print(f"[GPU {device_index}] Total Memory:           {total_mem:.2f} MB")
        print(f"[GPU {device_index}] Allocated by PyTorch:   {allocated:.2f} MB")
        print(f"[GPU {device_index}] Reserved by PyTorch:    {reserved:.2f} MB")
        print(f"[GPU {device_index}] Free within reserved:   {free_inside_reserved:.2f} MB")

    def wan_sampling_step(self, model_inputs: Dict, timestep: torch.Tensor,
                         guidance_scale: float = 10.0) -> torch.Tensor:
        """Execute one sampling step with proper channel handling"""
        x = model_inputs['noise_latents']  
        context = model_inputs['text_context']
        clip_fea = model_inputs['clip_context'] 
        y = model_inputs.get('conditional_latents', None)
        
        # Calculate context_lens
        # context_lens = torch.tensor([context.shape[1]], dtype=torch.long, device=context.device)
        
        x_input = []
        y_input = []
        
        for x_tensor, y_tensor in zip(x, y):
            # x: 16 channels noise + 4 channels padding = 20 channels
            x_16 = x_tensor[:16]
            padding = torch.zeros(4, *x_16.shape[1:], dtype=x_16.dtype, device=x_16.device)
            x_20 = torch.cat([x_16, padding], dim=0)  # 20 channels
            
            # y: 16 channels conditioning
            y_16 = y_tensor[:16]  # 16 channels
            
            x_input.append(x_20)
            y_input.append(y_16)
        
        seq_len = int(x_input[0].shape[1] * x_input[0].shape[2] * x_input[0].shape[3] * 1.5)
        
        # # Convert inputs to float32
        # x_input = [xi.to(torch.float32) for xi in x_input]
        # if y_input:
        #     y_input = [yi.to(torch.float32) for yi in y_input]
        # # context = context.to(torch.float32)
        # clip_fea = clip_fea.to(torch.float32)
        # timestep = timestep.to(torch.float32)
        
        # No autocast for float32
        model_output = self.model(
            x=x_input,
            t=timestep,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y_input,
            # context_lens=context_lens
        )
        
        # Convert output back to original precision if needed
        # if isinstance(model_output, list):
        #     model_output = [mo.to(self.param_dtype) for mo in model_output]
        # else:
        #     model_output = model_output.to(self.param_dtype)
        # Ensure output is a list
        if not isinstance(model_output, list):
            model_output = [model_output]
        
        # Convert output back to original precision if needed
        model_output = [mo.to(self.param_dtype) for mo in model_output]
        
        return model_output

    def generate_framepack(
        self,
        input_prompt: str,
        img: Image.Image,
        total_second_length: float = 5.0,
        max_area: int = 720 * 1280,
        shift: float = 5.0,
        sample_solver: str = 'unipc',
        sampling_steps: int = 25,
        guide_scale: float = 10.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        use_teacache: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Generate long video using FramePack with multi-GPU support"""
        print('Starting FramePack generation with multi-GPU support...')
        
        # Initial cleanup
        print("Step 0: GPU cleanup...")
        self.clear_gpu_memory()
        
        # Show GPU memory status
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        # Preprocess image
        img_inputs = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        
        # Calculate dimensions
        h, w = img_inputs.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // 
            self.patch_size[1] * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // 
            self.patch_size[2] * self.patch_size[2]
        )
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        
        # Calculate sections using FramePack strategy
        total_sections, latent_paddings = self.framepack_sampler.calculate_sections(
            total_second_length, fps=30
        )
        
        # Initialize history tensors (CPU)
        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, lat_h, lat_w), 
            dtype=torch.float16,
            device='cpu'
        )
        history_pixels = None
        total_generated_latent_frames = 0
        
        # Setup random generator
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device='cpu')
        seed_g.manual_seed(seed)
        torch.cuda.empty_cache()
        self.log_gpu_memory(self.device4)
        # STEP 1: Text encoding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
            
        print("Step 1: Text encoding...")
        inputs = self.tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True)
        n_inputs = self.tokenizer(n_prompt, return_tensors="pt", padding=True, truncation=True)
        
        self.text_encoder.to(self.device4)
        with torch.no_grad():
            context = self.text_encoder(**inputs.to(self.device4)).last_hidden_state.cpu()
            context_null = self.text_encoder(**n_inputs.to(self.device4)).last_hidden_state.cpu()
        
        self.text_encoder.cpu()
        torch.cuda.empty_cache()
        
        # STEP 2: CLIP encoding
        print("Step 2: CLIP encoding...")
        inputs = self.clip_processor(images=img, return_tensors="pt")
        
        self.clip_model.to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(pixel_values=inputs['pixel_values'].to(self.device))
            clip_features = outputs.image_embeds
        
        expected_dim = 1280
        if clip_features.shape[-1] != expected_dim:
            if not hasattr(self, 'clip_projection'):
                self.clip_projection = torch.nn.Linear(
                    clip_features.shape[-1], 
                    expected_dim, 
                    device=self.device, 
                    dtype=torch.float16
                )
            clip_features = self.clip_projection(clip_features)
            if clip_features.dim() == 2:
                clip_features = clip_features.unsqueeze(1)
        
        clip_context = clip_features.cpu()
        
        self.clip_model.cpu()
        if hasattr(self, 'clip_projection'):
            self.clip_projection.cpu()
        torch.cuda.empty_cache()
        
        # STEP 3: VAE encoding
        print("Step 3: VAE encoding...")
        img_input = img_inputs.unsqueeze(0).unsqueeze(2)
        
        self.vae.to(self.device)
        with torch.no_grad():
            latent_dist = self.vae.encode(img_input)
            start_latent = latent_dist.latent_dist.sample().cpu()
        
        self.vae.cpu()
        torch.cuda.empty_cache()
        
        clean_latents_pre = start_latent.to(dtype=history_latents.dtype)

        # Initialize sampling scheduler
        if sample_solver == 'unipc':
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=shift
            )
        else:
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=shift
            )
        
        scheduler.set_timesteps(sampling_steps, device=self.device)
        timesteps = scheduler.timesteps

        # Progressive generation loop
        for section_idx, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            
            print(f"\n=== Section {section_idx + 1}/{total_sections} ===")
            
            # Prepare clean latents
            target_shape = start_latent.shape[-2:]
            
            min_required_frames = 1 + 2 + 16
            if history_latents.shape[2] < min_required_frames:
                padding_needed = min_required_frames - history_latents.shape[2]
                padding = torch.zeros(
                    1, 16, padding_needed, *target_shape,
                    dtype=history_latents.dtype,
                    device='cpu'
                )
                history_latents = torch.cat([padding, history_latents], dim=2)

            clean_latents_post, clean_latents_2x, clean_latents_4x = \
                history_latents[:, :, :min_required_frames, :, :].split([1, 2, 16], dim=2)

            def resize_if_needed(tensor, target_shape):
                if tensor.shape[-2:] != target_shape:
                    orig_shape = tensor.shape
                    tensor_flat = tensor.flatten(0, 2)
                    tensor_flat_fp32 = tensor_flat.to(torch.float32)
                    tensor_resized = torch.nn.functional.interpolate(
                        tensor_flat_fp32.unsqueeze(1),
                        size=target_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    tensor_resized = tensor_resized.to(tensor.dtype)
                    return tensor_resized.unflatten(0, orig_shape[:-2])
                return tensor

            clean_latents_post = resize_if_needed(clean_latents_post, target_shape)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # Generate noise
            noise = torch.randn(
                1, 16, self.framepack_sampler.latent_window_size,
                lat_h, lat_w,
                dtype=torch.float16,
                generator=torch.Generator(device=self.device).manual_seed(seed + section_idx),
                device=self.device2
            )
            
            # For multi-GPU, model is already distributed
            if not self.use_multi_gpu:
                # Single GPU path
                print("Step 5: Moving main model to GPU...")
                try:
                    self.model.to(self.device)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"FAILED to move main model to GPU: {e}")
                    raise
            
            # Initialize TeaCache if enabled
            if use_teacache and hasattr(self.model, 'initialize_teacache'):
                self.model.initialize_teacache(enable_teacache=True, num_steps=sampling_steps)
            
            # Prepare inputs
            start_latent_gpu = start_latent.squeeze(2).to(self.device3)
            clean_latents_gpu = clean_latents.to(self.device3)
            
            noise_list = self.prepare_wan_noise_inputs(noise)
            y_list = self.prepare_wan_conditional_inputs(
                start_latent_gpu,
                reference_noise=noise_list
            )
            
            print("Step 6: Starting denoising loop...")
            
            # Denoising loop
            latents = noise
            for step_idx, t in enumerate(timesteps):
                if step_idx % 5 == 0:
                    print(f"  Denoising step {step_idx + 1}/{len(timesteps)}")
                
                # Move context to GPU
                context_gpu = context.to(self.device)
                context_null_gpu = context_null.to(self.device)
                clip_context_gpu = clip_context.to(self.device)
                
                # Prepare model inputs
                model_inputs = {
                    'noise_latents': self.prepare_wan_noise_inputs(latents),
                    'text_context': context_gpu,
                    'clip_context': clip_context_gpu,
                    'conditional_latents': y_list
                }
                
                # Classifier-free guidance
                if guide_scale > 1.0:
                    # Conditional prediction
                    noise_pred_cond = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                    
                    # Unconditional prediction
                    model_inputs['text_context'] = context_null_gpu
                    noise_pred_uncond = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                    
                    # Apply guidance
                    noise_pred = []
                    for cond, uncond in zip(noise_pred_cond, noise_pred_uncond):
                        guided = uncond + guide_scale * (cond - uncond)
                        noise_pred.append(guided)
                else:
                    noise_pred = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                
                # Scheduler step
                if isinstance(noise_pred, list):
                    noise_pred_batch = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred_batch = noise_pred
                    
                latents = scheduler.step(noise_pred_batch, t, latents).prev_sample
                
                # Clear intermediate tensors
                del context_gpu, context_null_gpu, clip_context_gpu
                if step_idx % 10 == 0:  # Periodic cleanup
                    torch.cuda.empty_cache()
            
            # Move model off GPU if single GPU mode
            if not self.use_multi_gpu:
                print("Step 7: Moving main model off GPU...")
                self.model.cpu()
                torch.cuda.empty_cache()
            
            # Move generated latents to CPU
            generated_latents = latents.cpu()
            
            if is_last_section:
                start_latent_cpu = start_latent.squeeze(2)
                generated_latents = torch.cat([start_latent_cpu, generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            
            # Update history
            generated_latents_for_history = generated_latents.to(dtype=history_latents.dtype)
            history_latents = torch.cat([generated_latents_for_history, history_latents], dim=2)

            # VAE decoding
            print("Step 8: VAE decoding...")
            torch.cuda.empty_cache()
            
            self.vae.to(self.device)
            
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :].to(self.device)

            if history_pixels is None:
                with torch.no_grad():
                    history_pixels = self.vae.decode(real_history_latents).sample.cpu()
            else:
                section_latent_frames = (self.framepack_sampler.latent_window_size * 2 + 1) if is_last_section else (self.framepack_sampler.latent_window_size * 2)
                overlapped_frames = self.framepack_sampler.latent_window_size * 4 - 3

                with torch.no_grad():
                    current_pixels = self.vae.decode(real_history_latents[:, :, :section_latent_frames]).sample.cpu()
                    history_pixels = self.framepack_sampler.soft_append_pixels(
                        current_pixels, history_pixels, overlapped_frames
                    )

            self.vae.cpu()
            torch.cuda.empty_cache()

            if progress_callback:
                generated_frames = max(0, total_generated_latent_frames * 4 - 3)
                total_seconds = max(0, generated_frames / 30)
                progress_callback({
                    'section': section_idx + 1,
                    'total_sections': total_sections,
                    'generated_frames': generated_frames,
                    'total_seconds': total_seconds,
                    'status': f'Completed section {section_idx + 1}/{total_sections}'
                })

            if is_last_section:
                break
        
        # Final cleanup
        print("Final cleanup...")
        self.clear_gpu_memory()
        
        return history_pixels

    def save_video_mp4(self, pixels: torch.Tensor, output_path: str, fps: int = 30, crf: int = 16):
        """Save generated video as MP4 file"""
        import cv2
        import tempfile
        import os
        
        # Convert tensor to numpy
        pixels = pixels.squeeze(0)  # Remove batch dimension
        pixels = (pixels + 1.0) / 2.0  # Denormalize
        pixels = (pixels * 255).clamp(0, 255).byte()
        pixels = pixels.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
        
        # Setup video writer
        height, width = pixels.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in pixels:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        
        # Re-encode with better compression if needed
        if crf < 16:
            temp_path = output_path.replace('.mp4', '_temp.mp4')
            os.rename(output_path, temp_path)
            
            cmd = f'ffmpeg -i {temp_path} -c:v libx264 -crf {crf} -pix_fmt yuv420p {output_path}'
            os.system(cmd)
            os.remove(temp_path)


class WanFramePackDemo:
    """Demo class showing how to use WAN with FramePack integration"""
    
    def __init__(self, checkpoint_dir: str, device_id: int = 0, use_multi_gpu: bool = True):
        self.config = WanFramePackConfig()
        self.model = WanI2VFramePack(
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            use_quantization='fp16',
            use_multi_gpu=use_multi_gpu
        )
    
    def generate_long_video(self, image_path: str, prompt: str, duration: float = 60.0) -> str:
        """Generate a long video using FramePack"""
        img = Image.open(image_path).convert('RGB')
        
        def progress_callback(info):
            if 'status' in info:
                print(info['status'])
            if 'section' in info and 'total_sections' in info:
                section_progress = (info['section'] / info['total_sections']) * 100
                print(f"Overall progress: {section_progress:.1f}% - "
                      f"Generated {info.get('generated_frames', 0)} frames "
                      f"({info.get('total_seconds', 0):.1f}s)")
        
        pixels = self.model.generate_framepack(
            input_prompt=prompt,
            img=img,
            total_second_length=duration,
            sampling_steps=25,
            guide_scale=10.0,
            use_teacache=True,
            progress_callback=progress_callback
        )
        
        # Save video
        output_path = f"long_video_{duration}s.mp4"
        self.model.save_video_mp4(pixels, output_path, fps=30)
        return output_path


def test_wan_framepack_multigpu():
    """Test function with multi-GPU support"""
    
    # Set environment for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Example configuration
    checkpoint_dir = "downloads/Wan2_1-I2V-ATI-14B_fp16.safetensors"
    
    try:
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} - {props.total_memory/1024**3:.1f}GB")
        
        # Initialize demo with multi-GPU support
        use_multi_gpu = num_gpus > 1
        demo = WanFramePackDemo(checkpoint_dir, device_id=0, use_multi_gpu=use_multi_gpu)
        
        # Test long video generation with FramePack
        print("\nTesting long video generation with FramePack...")
        long_video = demo.generate_long_video(
            image_path="random-image.jpg", 
            prompt="A boy dancing gracefully with clear movements",
            duration=30.0
        )
        print(f"Generated long video: {long_video}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run multi-GPU test
    test_wan_framepack_multigpu()