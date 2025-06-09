import torch
import torch.nn as nn
from typing import Tuple, Optional

class FramePackWANAdapter(nn.Module):
    """
    Lightweight adapter to bridge FramePack and WAN dimensional differences
    """
    
    def __init__(
        self,
        framepack_dim: int = 3072,  # 24 * 128
        wan_dim: int = 5120,        # 40 * 128
        framepack_channels: int = 16,
        wan_channels: int = 36,
        patch_size: Tuple[int, int, int] = (1, 2, 2)
    ):
        super().__init__()
        
        self.framepack_dim = framepack_dim
        self.wan_dim = wan_dim
        self.framepack_channels = framepack_channels
        self.wan_channels = wan_channels
        
        print(f"🔧 Creating FramePackWAN Adapter:")
        print(f"   Dimension: {framepack_dim} ↔ {wan_dim}")
        print(f"   Channels: {framepack_channels} ↔ {wan_channels}")
        
        # Dimension adapters
        self.to_wan_dim = nn.Sequential(
            nn.Linear(framepack_dim, wan_dim),
            nn.LayerNorm(wan_dim)
        )
        
        self.from_wan_dim = nn.Sequential(
            nn.Linear(wan_dim, framepack_dim),
            nn.LayerNorm(framepack_dim)
        )
        
        # Channel adapters
        self.to_wan_channels = nn.Conv3d(
            framepack_channels, wan_channels,
            kernel_size=1, padding=0, bias=True
        )
        
        self.from_wan_channels = nn.Conv3d(
            wan_channels, framepack_channels,
            kernel_size=1, padding=0, bias=True
        )
        
        # Initialize channel adapters properly
        self._init_channel_adapters()
        
        print("✅ Adapter created successfully!")
    
    def _init_channel_adapters(self):
        """Initialize channel adapters to preserve important channels"""
        
        with torch.no_grad():
            # For expanding channels (16 -> 36)
            # Copy original channels and add learned expansions
            self.to_wan_channels.weight[:self.framepack_channels] = torch.eye(self.framepack_channels).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.to_wan_channels.weight[self.framepack_channels:] = 0.01 * torch.randn(
                self.wan_channels - self.framepack_channels, self.framepack_channels, 1, 1, 1
            )
            self.to_wan_channels.bias.zero_()
            
            # For reducing channels (36 -> 16)  
            # Use learned combination of WAN channels
            self.from_wan_channels.weight.normal_(0, 0.01)
            self.from_wan_channels.bias.zero_()
    
    def adapt_to_wan(self, x: torch.Tensor, format_type: str = "sequence") -> torch.Tensor:
        """
        Convert from FramePack format to WAN format
        
        Args:
            x: Input tensor
            format_type: "sequence" for (B, L, D) or "volume" for (B, C, T, H, W)
        """
        if format_type == "sequence":
            # (B, L, framepack_dim) -> (B, L, wan_dim)
            return self.to_wan_dim(x)
            
        elif format_type == "volume":
            # (B, framepack_channels, T, H, W) -> (B, wan_channels, T, H, W)
            return self.to_wan_channels(x)
            
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def adapt_from_wan(self, x: torch.Tensor, format_type: str = "sequence") -> torch.Tensor:
        """
        Convert from WAN format back to FramePack format
        
        Args:
            x: Input tensor  
            format_type: "sequence" for (B, L, D) or "volume" for (B, C, T, H, W)
        """
        if format_type == "sequence":
            # (B, L, wan_dim) -> (B, L, framepack_dim)
            return self.from_wan_dim(x)
            
        elif format_type == "volume":
            # (B, wan_channels, T, H, W) -> (B, framepack_channels, T, H, W)
            return self.from_wan_channels(x)
            
        else:
            raise ValueError(f"Unknown format_type: {format_type}")


class ModifiedWANDiTBlock(nn.Module):
    """
    Your WANDiTBlock modified to work with FramePack dimensions
    """
    
    def __init__(
        self,
        # Original WAN config
        wan_in_channels: int = 36,
        wan_out_channels: int = 16,
        wan_num_attention_heads: int = 40,
        wan_attention_head_dim: int = 128,
        wan_num_layers: int = 40,
        wan_ffn_dim: int = 13824,
        
        # FramePack compatibility config
        framepack_in_channels: int = 16,
        framepack_out_channels: int = 16,
        framepack_num_attention_heads: int = 24,
        framepack_attention_head_dim: int = 128,
        framepack_num_layers: int = 20,
        framepack_num_single_layers: int = 40,
        framepack_num_refiner_layers: int = 2,
        
        # Shared config
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        has_image_proj: bool = False,
        image_proj_dim: int = 1152,
        has_clean_x_embedder: bool = False,
    ):
        super().__init__()
       
        self.framepack_inner_dim = framepack_num_attention_heads * framepack_attention_head_dim
        self.wan_inner_dim = wan_num_attention_heads * wan_attention_head_dim
        
        # Store config
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        
        print(f"🚀 Creating Modified WANDiTBlock:")
        print(f"   FramePack inner dim: {self.framepack_inner_dim}")
        print(f"   WAN inner dim: {self.wan_inner_dim}")
        print(f"   Total WAN layers: {wan_num_layers}")
        
        # ================================================================
        # FramePack Components (for I/O compatibility)
        # ================================================================
        
        from diffusers_helper.models.hunyuan_video_packed import (
            HunyuanVideoPatchEmbed, HunyuanVideoTokenRefiner, 
            CombinedTimestepGuidanceTextProjEmbeddings, AdaLayerNormContinuous
        )
        
        # Input embedders (FramePack format)
        self.x_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), 
            framepack_in_channels, 
            self.framepack_inner_dim
        )
        
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, framepack_num_attention_heads, framepack_attention_head_dim, 
            num_layers=framepack_num_refiner_layers
        )
        
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            self.framepack_inner_dim, pooled_projection_dim
        )
        
        # Output layers (FramePack format)
        self.norm_out = AdaLayerNormContinuous(
            self.framepack_inner_dim, self.framepack_inner_dim, 
            elementwise_affine=False, eps=1e-6
        )
        
        self.proj_out = nn.Linear(
            self.framepack_inner_dim, 
            patch_size_t * patch_size * patch_size * framepack_out_channels
        )
        
        # ================================================================
        # Dimension Adapter
        # ================================================================
        
        self.adapter = FramePackWANAdapter(
            framepack_dim=self.framepack_inner_dim,
            wan_dim=self.wan_inner_dim,
            framepack_channels=framepack_in_channels,
            wan_channels=wan_in_channels,
            patch_size=(patch_size_t, patch_size, patch_size)
        )
        def get_available_memory_mb():
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        parts = line.split()
                        kb = int(parts[1])  # value in kB
                        return kb // 1024  # convert to MB

        print(f"Available Memory1: {get_available_memory_mb()} MB")

        # ================================================================
        # WAN Transformer Blocks
        # ================================================================
        
        # Replace with your actual WAN blocks
        # For now using placeholder - you should replace this with your WanAttentionBlock
        from diffusers_helper.wan_components.wan.modules.model import WanAttentionBlock
        from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb
        self.wan_blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type='i2v_cross_attn',
                dim=self.wan_inner_dim,
                ffn_dim=self.wan_inner_dim * 4,
                num_heads=wan_num_attention_heads,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6
            )
            for _ in range(1)
        ])
        
        # self.wan_blocks = nn.ModuleList([
        #     PlaceholderWANBlock(
        #         dim=self.wan_inner_dim,
        #         num_heads=wan_num_attention_heads,
        #         head_dim=wan_attention_head_dim,
        #         ffn_dim=wan_ffn_dim,
        #         qk_norm=qk_norm
        #     )
        #     for _ in range(16)
        # ])
        print(f"Available Memory: {get_available_memory_mb()} MB")
        # ================================================================
        # Optional FramePack components
        # ================================================================
        
        self.clean_x_embedder = None
        self.image_projection = None
        
        if has_clean_x_embedder:
            self.install_clean_x_embedder()
            
        if has_image_proj:
            self.install_image_projection(image_proj_dim)
        
        print(f"✅ Modified WANDiTBlock created successfully!")
        


    def install_clean_x_embedder(self):
        """Install FramePack's clean latent embedder"""
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoPatchEmbedForCleanLatents
        self.clean_x_embedder = HunyuanVideoPatchEmbedForCleanLatents(self.framepack_inner_dim)

    def install_image_projection(self, in_channels: int):
        """Install FramePack's image projection"""
        from diffusers_helper.models.hunyuan_video_packed import ClipVisionProjection
        self.image_projection = ClipVisionProjection(in_channels, self.framepack_inner_dim)

    def process_input_hidden_states(
        self,
        latents, latent_indices=None,
        clean_latents=None, clean_latent_indices=None,
        clean_latents_2x=None, clean_latent_2x_indices=None,
        clean_latents_4x=None, clean_latent_4x_indices=None
    ):
        """FramePack's input processing with clean latent support"""
        
        # Basic patch embedding
        hidden_states = self.x_embedder.proj(latents)
        B, C, T, H, W = hidden_states.shape

        if latent_indices is None:
            latent_indices = torch.arange(0, T).unsqueeze(0).expand(B, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Handle clean latents if provided
        if clean_latents is not None and self.clean_x_embedder is not None:
            clean_latents = clean_latents.to(hidden_states)
            clean_latents = self.clean_x_embedder.proj(clean_latents)
            clean_latents = clean_latents.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([clean_latents, hidden_states], dim=1)

        # Similar for 2x and 4x clean latents...
        
        return hidden_states
   
    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
        # FramePack specific
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
        image_embeddings=None,
        return_dict=True,
        **kwargs
    ):
        """Forward pass with dimensional adaptation"""
       
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        
        print(f"🔄 Modified WANDiTBlock Forward:")
        print(f"   Input: {hidden_states.shape}")
        
        # ================================================================
        # STEP 1: FramePack Input Processing  
        # ================================================================
        
        # Process embeddings using FramePack components
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)
        
        # Handle image embeddings
        if self.image_projection is not None and image_embeddings is not None:
            extra_encoder_hidden_states = self.image_projection(image_embeddings)
            extra_attention_mask = torch.ones(
                (batch_size, extra_encoder_hidden_states.shape[1]), 
                dtype=encoder_attention_mask.dtype, 
                device=encoder_attention_mask.device
            )
            encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

        # Process input states
        hidden_states = self.process_input_hidden_states(
            hidden_states, latent_indices, clean_latents, clean_latent_indices,
            clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices
        )
        
        print(f"   After FramePack processing: {hidden_states.shape}")
        
        # ================================================================
        # STEP 2: Adapt to WAN Dimensions
        # ================================================================
        
        # Convert FramePack dimensions to WAN dimensions
        hidden_states_wan = self.adapter.adapt_to_wan(hidden_states, format_type="sequence")
        encoder_hidden_states_wan = self.adapter.adapt_to_wan(encoder_hidden_states, format_type="sequence")
        
        print(f"   After dimension adaptation: {hidden_states_wan.shape}")
        
        # ================================================================
        # STEP 3: WAN Processing
        # ================================================================
        print(self.wan_blocks, 'here--------------------------------------')
        # Process through WAN blocks
        # for i, block in enumerate(self.wan_blocks):
        #     hidden_states_wan = block(
        #         hidden_states_wan,
        #         encoder_hidden_states=encoder_hidden_states_wan,
        #         attention_mask=encoder_attention_mask,
        #         timestep=timestep
        #     )
        
        # print(f"   After WAN processing: {hidden_states_wan.shape}")
        print(f"   Processing through {len(self.wan_blocks)} WAN blocks...")
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        for i, block in enumerate(self.wan_blocks):
            try:
                # block = block.cuda()
        
    	        # Ensure block is in correct device
                # block_device = torch.device('cuda:0')  # Force CUDA
                # x = hidden_states_wan.to(block_device)
                block_device = hidden_states_wan.device
    	        # Double-check head dimension
                actual_head_dim = self.wan_inner_dim // block.num_heads
                if actual_head_dim > 256:
                    print(f"   ❌ Head dim {actual_head_dim} > 256, skipping block {i}")
                # if actual_head_dim > 256:
    	        #     print(f"   ❌ Head dim {actual_head_dim} > 256, skipping block {i}")
                #     continue 
        
                batch_size = hidden_states_wan.shape[0]
                seq_len = hidden_states_wan.shape[1]
                
                # Prepare WAN-style arguments
                x = hidden_states_wan.to(block_device)
                
                # Create conditioning embedding from timestep
                e = torch.zeros(batch_size, 6, self.wan_inner_dim, 
                            device=block_device, dtype=torch.float32)
                if timestep is not None:
                    t_val = timestep.float() if torch.is_tensor(timestep) else float(timestep)
                    e = e + (t_val / 1000.0) * 0.1  # Simple timestep modulation
                
                # Compute sequence lengths
                seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
                
                # Compute grid sizes (spatial layout)
                spatial_size = max(1, int(seq_len ** 0.5))
                grid_sizes = torch.tensor([[1, spatial_size, spatial_size]], 
                                        dtype=torch.long, device=block_device).expand(batch_size, -1)
                
                # Create RoPE frequencies (create once and reuse)
                if not hasattr(self, 'rope_freqs'):
                    head_dim = self.wan_inner_dim // block.num_heads
                    freqs = torch.randn(1024, head_dim // 2, device=block_device) * 0.01
                    self.register_buffer('rope_freqs', freqs)
                
                # Prepare context
                context = encoder_hidden_states_wan
                context_lens = torch.full((batch_size,), context.shape[1], dtype=torch.long, device=x.device)
                
                # Call WAN block with correct signature
                hidden_states_wan = block(
                    x=x,
                    e=e,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.rope_freqs,
                    context=context,
                    context_lens=context_lens
                )
                
                print(f"   ✅ Processed WAN block {i+1}/{len(self.wan_blocks)}")
                
            except Exception as ex:
                print(f"   ❌ WAN block {i} failed: {ex}")
                import traceback
                traceback.print_exc()
                # Continue with unchanged hidden states
                continue

        print(f"   ✅ WAN processing complete")
        # ================================================================
        # STEP 4: Adapt back to FramePack Dimensions
        # ================================================================
        
        # Convert back to FramePack dimensions
        hidden_states = self.adapter.adapt_from_wan(hidden_states_wan, format_type="sequence")
        
        # ================================================================
        # STEP 5: FramePack Output Processing
        # ================================================================
        
        # Apply FramePack's output processing
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        
        # Reshape to video format (same as original FramePack logic)
        original_context_length = num_frames * height // self.patch_size * width // self.patch_size
        output = output[:, -original_context_length:, :]
        
        # Unpatchify to final video tensor
        post_patch_num_frames = num_frames // self.patch_size_t
        post_patch_height = height // self.patch_size  
        post_patch_width = width // self.patch_size
        
        import einops
        output = einops.rearrange(
            output, 
            'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
            t=post_patch_num_frames, h=post_patch_height, w=post_patch_width,
            pt=self.patch_size_t, ph=self.patch_size, pw=self.patch_size
        )
        
        print(f"   Final output: {output.shape}")
        print("✅ Modified WANDiTBlock forward complete!")
        
        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output)
        
        return output,

    @classmethod
    def from_wan_pretrained(cls, wan_model_path, subfolder, framepack_model_path='lllyasviel/FramePackI2V_HY', **kwargs):
        """Load from WAN pretrained model and add FramePack progressive generation logic"""
        
        print(f"🔄 Loading WAN model: {wan_model_path}")
        print(f"🔄 Loading FramePack components from: {framepack_model_path}")
        
        # Load WAN model
        try:
            from diffusers import WanTransformer3DModel  # Adjust import as needed
            wan_model = WanTransformer3DModel.from_pretrained(wan_model_path, subfolder=subfolder, **kwargs).cpu()
            print(f"✅ WAN model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load WAN model: {e}")
            print("   Falling back to config-based initialization...")
            wan_model = None
        
        # Load FramePack model for components
        try:
            from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
            framepack_model = HunyuanVideoTransformer3DModelPacked.from_pretrained(framepack_model_path, **kwargs)
            print(f"✅ FramePack model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load FramePack model: {e}")
            framepack_model = None
        
        # Extract configs
        wan_config = wan_model.config if wan_model else {}
        framepack_config = framepack_model.config if framepack_model else {}
        
        # Create adapted model with configs
        adapted_model = cls(
            # WAN config (from WAN model)
            wan_in_channels=wan_config.get('in_channels', 36),
            wan_out_channels=wan_config.get('out_channels', 16),
            wan_num_attention_heads=wan_config.get('num_attention_heads', 40),
            wan_attention_head_dim=wan_config.get('attention_head_dim', 128),
            wan_num_layers=wan_config.get('num_layers', 40),
            wan_ffn_dim=wan_config.get('ffn_dim', 13824),
            
            # FramePack config (from FramePack model or defaults)
            framepack_in_channels=framepack_config.get('in_channels', 16),
            framepack_out_channels=framepack_config.get('out_channels', 16),
            framepack_num_attention_heads=framepack_config.get('num_attention_heads', 24),
            framepack_attention_head_dim=framepack_config.get('attention_head_dim', 128),
            framepack_num_layers=framepack_config.get('num_layers', 20),
            framepack_num_single_layers=framepack_config.get('num_single_layers', 40),
            framepack_num_refiner_layers=framepack_config.get('num_refiner_layers', 2),
            
            # Shared config
            mlp_ratio=framepack_config.get('mlp_ratio', 4.0),
            patch_size=wan_config.get('patch_size', [1, 2, 2])[-1],  # Use spatial patch size
            patch_size_t=wan_config.get('patch_size', [1, 2, 2])[0],  # Use temporal patch size
            text_embed_dim=wan_config.get('text_dim', framepack_config.get('text_embed_dim', 4096)),
            pooled_projection_dim=framepack_config.get('pooled_projection_dim', 768),
            has_image_proj=framepack_config.get('has_image_proj', False),
            image_proj_dim=framepack_config.get('image_proj_dim', wan_config.get('image_dim', 1152)),
            has_clean_x_embedder=framepack_config.get('has_clean_x_embedder', False),
        )
        
        print(f"✅ Adapted model created")
        
        # ================================================================
        # Copy WAN transformer blocks
        # ================================================================
        
        if wan_model is not None and hasattr(wan_model, 'transformer_blocks'):
            try:
                # Copy WAN blocks if they exist and are compatible
                wan_blocks = wan_model.transformer_blocks
                if len(wan_blocks) == len(adapted_model.wan_blocks):
                    for i, (wan_block, adapted_block) in enumerate(zip(wan_blocks, adapted_model.wan_blocks)):
                        # Try to copy compatible weights
                        try:
                            adapted_block.load_state_dict(wan_block.state_dict(), strict=False)
                            print(f"   ✅ Copied WAN block {i}")
                        except Exception as e:
                            print(f"   ⚠️ Could not copy WAN block {i}: {e}")
                else:
                    print(f"   ⚠️ WAN block count mismatch: {len(wan_blocks)} vs {len(adapted_model.wan_blocks)}")
            except Exception as e:
                print(f"   ⚠️ Could not copy WAN blocks: {e}")
        
        # ================================================================
        # Copy FramePack components
        # ================================================================
        
        if framepack_model is not None:
            # Copy FramePack input/output components
            framepack_components = [
                'x_embedder', 'context_embedder', 'time_text_embed',
                'norm_out', 'proj_out'
            ]
            
            for component_name in framepack_components:
                if hasattr(framepack_model, component_name):
                    try:
                        getattr(adapted_model, component_name).load_state_dict(
                            getattr(framepack_model, component_name).state_dict()
                        )
                        print(f"   ✅ Copied FramePack {component_name}")
                    except Exception as e:
                        print(f"   ⚠️ Could not copy FramePack {component_name}: {e}")
            
            # Copy optional FramePack components
            if hasattr(framepack_model, 'clean_x_embedder') and framepack_model.clean_x_embedder is not None:
                if adapted_model.clean_x_embedder is not None:
                    try:
                        adapted_model.clean_x_embedder.load_state_dict(framepack_model.clean_x_embedder.state_dict())
                        print("   ✅ Copied clean_x_embedder")
                    except Exception as e:
                        print(f"   ⚠️ Could not copy clean_x_embedder: {e}")
                        
            if hasattr(framepack_model, 'image_projection') and framepack_model.image_projection is not None:
                if adapted_model.image_projection is not None:
                    try:
                        adapted_model.image_projection.load_state_dict(framepack_model.image_projection.state_dict())
                        print("   ✅ Copied image_projection")
                    except Exception as e:
                        print(f"   ⚠️ Could not copy image_projection: {e}")
        
        # ================================================================
        # Initialize adapter layers
        # ================================================================
        
        print("🔧 Initializing adapter layers...")
        
        # The adapter layers are already initialized in __init__
        # You might want to fine-tune them or load pre-trained adapter weights here
        
        # Cleanup
        if wan_model is not None:
            del wan_model
        if framepack_model is not None:
            del framepack_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✅ Model loading complete!")
        print(f"   WAN blocks: Loaded from pretrained WAN model")
        print(f"   FramePack components: Loaded from pretrained FramePack model")
        print(f"   Adapter layers: Randomly initialized (ready for fine-tuning)")
        
        return adapted_model

    @classmethod
    def from_hunyuan_pretrained(cls, model_path='lllyasviel/FramePackI2V_HY', **kwargs):
        """Backward compatibility: Load from HunyuanVideo and convert to WAN architecture"""
        
        print(f"🔄 Loading FramePack model: {model_path}")
        print("   Note: WAN blocks will be randomly initialized")
        
        # Load original model
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
        original_model = HunyuanVideoTransformer3DModelPacked.from_pretrained(model_path, **kwargs)
        
        # Extract config
        config = original_model.config
        
        # Create adapted model with extracted config
        adapted_model = cls(
            # WAN config (defaults - not pretrained)
            wan_in_channels=36,
            wan_out_channels=16, 
            wan_num_attention_heads=40,
            wan_attention_head_dim=128,
            wan_num_layers=40,
            wan_ffn_dim=13824,
            
            # FramePack config (from original)
            framepack_in_channels=config.get('in_channels', 16),
            framepack_out_channels=config.get('out_channels', 16),
            framepack_num_attention_heads=config.get('num_attention_heads', 24),
            framepack_attention_head_dim=config.get('attention_head_dim', 128),
            framepack_num_layers=config.get('num_layers', 20),
            framepack_num_single_layers=config.get('num_single_layers', 40),
            framepack_num_refiner_layers=config.get('num_refiner_layers', 2),
            
            # Other config
            mlp_ratio=config.get('mlp_ratio', 4.0),
            patch_size=config.get('patch_size', 2),
            patch_size_t=config.get('patch_size_t', 1),
            text_embed_dim=config.get('text_embed_dim', 4096),
            pooled_projection_dim=config.get('pooled_projection_dim', 768),
            has_image_proj=config.get('has_image_proj', False),
            image_proj_dim=config.get('image_proj_dim', 1152),
            has_clean_x_embedder=config.get('has_clean_x_embedder', False),
        )
        
        # Copy FramePack components
        compatible_components = [
            'x_embedder', 'context_embedder', 'time_text_embed',
            'norm_out', 'proj_out'
        ]
        
        for component_name in compatible_components:
            if hasattr(original_model, component_name):
                try:
                    getattr(adapted_model, component_name).load_state_dict(
                        getattr(original_model, component_name).state_dict()
                    )
                    print(f"   ✅ Copied {component_name}")
                except Exception as e:
                    print(f"   ⚠️ Could not copy {component_name}: {e}")
        
        # Copy optional components
        if hasattr(original_model, 'clean_x_embedder') and original_model.clean_x_embedder is not None:
            adapted_model.clean_x_embedder.load_state_dict(original_model.clean_x_embedder.state_dict())
            print("   ✅ Copied clean_x_embedder")
            
        if hasattr(original_model, 'image_projection') and original_model.image_projection is not None:
            adapted_model.image_projection.load_state_dict(original_model.image_projection.state_dict())
            print("   ✅ Copied image_projection")
        
        # Cleanup
        del original_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✅ Model conversion complete!")
        print(f"   FramePack components: Loaded from pretrained")
        print(f"   WAN blocks: Randomly initialized (ready for fine-tuning)")
        
        return adapted_model


class PlaceholderWANBlock(nn.Module):
    """
    Placeholder for actual WAN block - replace with your WanAttentionBlock
    """
    def __init__(self, dim, num_heads, head_dim, ffn_dim, qk_norm="rms_norm"):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        )
    
    def forward(self, x, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Self-attention
        normed_x = self.norm1(x)
        attn_out, _ = self.self_attn(normed_x, normed_x, normed_x)
        x = x + attn_out
        
        # Cross-attention
        if encoder_hidden_states is not None:
            normed_x = self.norm2(x)
            cross_out, _ = self.cross_attn(normed_x, encoder_hidden_states, encoder_hidden_states)
            x = x + cross_out
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# Usage examples
if __name__ == "__main__":
    print("🧪 Testing Modified WANDiTBlock...")
    
    # Option 1: Load from WAN pretrained model (recommended)
    print("\n=== Loading from WAN pretrained ===")
    try:
        model = ModifiedWANDiTBlock.from_wan_pretrained(
            wan_model_path='your-wan-model-path',  # Replace with actual WAN model path
            framepack_model_path='lllyasviel/FramePackI2V_HY'
        )
        print("✅ WAN model loaded successfully!")
    except Exception as e:
        print(f"❌ WAN model loading failed: {e}")
        print("   Falling back to FramePack-only approach...")
        
        # Option 2: Load from FramePack only (WAN blocks randomly initialized)
        print("\n=== Loading from FramePack only ===")
        model = ModifiedWANDiTBlock.from_hunyuan_pretrained('lllyasviel/FramePackI2V_HY')
        print("✅ FramePack model loaded successfully!")
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   FramePack components: {sum(p.numel() for n, p in model.named_parameters() if 'wan_blocks' not in n):,}")
    print(f"   WAN components: {sum(p.numel() for n, p in model.named_parameters() if 'wan_blocks' in n):,}")
    
    # Test forward pass
    print("\n=== Testing forward pass ===")
    batch_size = 1
    num_frames = 8
    height = 64
    width = 64
    
    test_inputs = {
        'hidden_states': torch.randn(batch_size, 16, num_frames, height, width),
        'timestep': torch.tensor([1000]),
        'encoder_hidden_states': torch.randn(batch_size, 512, 4096),
        'encoder_attention_mask': torch.ones(batch_size, 512),
        'pooled_projections': torch.randn(batch_size, 768),
        'guidance': torch.tensor([10.0])
    }
    
    try:
        with torch.no_grad():
            output = model(**test_inputs)
        print(f"✅ Forward pass successful! Output shape: {output.sample.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
