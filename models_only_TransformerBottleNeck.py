################################################################################
# TransformerBottleneck
################################################################################
class TransformerBottleneck(nn.Module):
    """
    Replaces the original bottleneck residual block with a small Transformer block.
    The idea:
      1) Project in_channels -> mid_channels (1x1 conv).
      2) Flatten spatial dims (H*W) as sequence length.
      3) Pass through nn.TransformerEncoderLayer (self-attention).
      4) Reshape back to (B, mid_channels, H, W).
      5) Optionally project out again if needed.
    """

    def __init__(self, in_channels=128, mid_channels=256, n_heads=4, dim_feedforward=1024):
        super().__init__()
        
        # 1) Project from in_channels -> mid_channels
        self.proj_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        
        # 2) A single TransformerEncoderLayer
        #    batch_first=True => (B, seq_len, d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=mid_channels, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        
        # 3) Optional: final 1x1 conv to stay at mid_channels
        self.proj_out = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        Returns: (B, mid_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 1) Project to mid_channels
        x = self.proj_in(x)     # (B, mid_channels, H, W)
        
        # 2) Reshape for Transformer: (B, H*W, mid_channels)
        #    batch_first=True => expects (B, seq_len, d_model)
        x = x.flatten(2)        # (B, mid_channels, H*W)
        x = x.permute(0, 2, 1)  # (B, H*W, mid_channels)
        
        # 3) Forward pass through TransformerEncoderLayer
        x = self.transformer(x) # (B, H*W, mid_channels)
        
        # 4) Reshape back to (B, mid_channels, H, W)
        x = x.permute(0, 2, 1)  # (B, mid_channels, H*W)
        x = x.view(B, -1, H, W) # (B, mid_channels, H, W)
        
        # 5) Final projection + ReLU
        x = self.proj_out(x)    # (B, mid_channels, H, W)
        x = self.relu(x)
        
        return x

# -------------------------------------------------------------------------
# ResidualConvBlock
# -------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    """
    A 2D residual block with:
      - two conv layers (3x3)
      - GroupNorm(1, C) after each conv
      - ReLU activation
      - optional shortcut if in_channels != out_channels
    This matches your updated version that uses GroupNorm to emulate LayerNorm behavior in CNNs.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        
        # First convolution (3x3), groupnorm, ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution (3x3), groupnorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(1, out_channels)
        
        # Shortcut / identity mapping
        # If in/out channels differ, use a 1x1 conv; otherwise do nothing.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward pass:
          1) Apply conv1 + norm1 + ReLU
          2) Apply conv2 + norm2
          3) Add the original 'x' (optionally projected) to the result
          4) ReLU again
        """
        identity = self.shortcut(x)  # Might be 1x1 conv if channels differ

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.relu(out)
        return out


# -------------------------------------------------------------------------
# EncoderBlock
# -------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    A UNet encoder block:
      1) A ResidualConvBlock for feature extraction
      2) A MaxPool2d for spatial downsampling
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        # A residual conv block to learn features
        self.conv_block = ResidualConvBlock(in_channels, out_channels)
        # 2x2 pooling to halve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Returns:
          f:  feature map after the conv block
          p:  pooled feature map for the next encoding stage
        """
        f = self.conv_block(x)  # (B, out_channels, H, W)
        p = self.pool(f)        # (B, out_channels, H/2, W/2)
        return f, p


# -------------------------------------------------------------------------
# DecoderBlock
# -------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    A UNet decoder block:
      1) ConvTranspose2d for upsampling
      2) Concatenate with skip features
      3) ResidualConvBlock to combine them
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # 2x upsampling via transposed convolution
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # After concatenation, the channel dimension is out_channels + skip_channels (= out_channels).
        # But "skip_channels" = out_channels in this standard pattern, so total is (out_channels * 2).
        self.conv_block = ResidualConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip_features):
        """
        Args:
          x: upsampled feature map
          skip_features: feature map from the encoder
        Returns:
          A combined feature map after upsampling, concatenation, and residual conv block.
        """
        x = self.conv_transpose(x)                   # (B, out_channels, 2H, 2W)
        x = torch.cat((x, skip_features), dim=1)     # (B, out_channels*2, 2H, 2W)
        x = self.conv_block(x)
        return x


# -------------------------------------------------------------------------
# EncoderUNet
# -------------------------------------------------------------------------
class EncoderUNet(nn.Module):
    """
    Encodes two vectors (each ~ [B, 104]) + multiple scalars (e.g. 5 scalars)
    into a single-channel 131×131 output image via a UNet-like architecture.
    
    Key points:
      - Each scalar is individually expanded to 16 dims via a separate Linear.
      - The two vectors are flattened and combined, then projected to 128 dims.
      - Combined dimension => 128 + (scalar_count*16).
      - Mapped to 1×64×64 "latent image".
      - Passes through 3-level encoder + bottleneck + 3-level decoder.
      - Upsamples from 64×64 to 128×128, then final resize to 131×131.
    """
    def __init__(self, vector_dim, scalar_count):
        super(EncoderUNet, self).__init__()
        
        # Activation for general use
        self.relu = nn.ReLU(inplace=True)
        
        # -----------------------------------------------------------------
        # 1) Vector processing
        #    - Expecting vector1, vector2 each ~ [B, 104]
        #    - Combine => [B, 2*vector_dim], project to [B, 128]
        # -----------------------------------------------------------------
        self.vector_fc = nn.Linear(vector_dim * 2, 128)
        self.vector_norm = nn.LayerNorm(128)  # optional normalization
        
        # -----------------------------------------------------------------
        # 2) Scalar processing
        #    - Each scalar => 1->16
        #    - total scalar_count => scalar_count*16
        # -----------------------------------------------------------------
        self.scalar_fc_list = nn.ModuleList([
            nn.Linear(1, 16) for _ in range(scalar_count)
        ])
        self.scalar_norm = nn.LayerNorm(scalar_count * 16)
        
        # -----------------------------------------------------------------
        # 3) Combine vector+scalar => feed to latent_to_image => 1×64×64
        # -----------------------------------------------------------------
        in_features = 128 + scalar_count * 16  # e.g. 128 + 5*16 = 128 + 80 = 208
        self.latent_to_image = nn.Linear(in_features, 64 * 64)  # => 4096 => (1,64,64)
        
        # -----------------------------------------------------------------
        # 4) UNet encoder
        # -----------------------------------------------------------------
        self.encoder1 = EncoderBlock(in_channels=1,  out_channels=32)   # 64x64 -> 32x32
        self.encoder2 = EncoderBlock(in_channels=32, out_channels=64)   # 32x32 -> 16x16
        self.encoder3 = EncoderBlock(in_channels=64, out_channels=128)  # 16x16 -> 8x8
        
        # Transformer-based Bottleneck (instead of a residual conv block)
        self.bottleneck = TransformerBottleneck(
            in_channels=128, 
            mid_channels=256, 
            n_heads=4, 
            dim_feedforward=1024
        )
        
        # -----------------------------------------------------------------
        # 5) UNet decoder
        # -----------------------------------------------------------------
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128) # 8x8  -> 16x16
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)  # 16x16 -> 32x32
        self.decoder1 = DecoderBlock(in_channels=64,  out_channels=32)  # 32x32 -> 64x64
        
        # Upsample from 64x64 -> 128x128
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                      kernel_size=3, stride=2, 
                                      padding=1, output_padding=1)
        
        # Final conv from 16->1 channel
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        
        # Resize 128x128 -> 131x131
        self.final_resize = nn.Upsample(size=(131, 131), mode='bilinear', align_corners=False)

    def forward(self, vector1, vector2, scalars):
        """
        Args:
          vector1: [B, 1, 104] (or similar) => Flatten to [B, 104]
          vector2: [B, 1, 104] => Flatten to [B, 104]
          scalars: [B, 1, scalar_count], e.g. [B,1,5] => Squeeze => [B,5]
        
        Returns:
          (B, 1, 131, 131) single-channel output image
        """
        # -----------------------------------------------------------------
        # Flatten vectors
        # -----------------------------------------------------------------
        B = vector1.size(0)
        vector1 = vector1.view(B, -1)  # => [B, 104]
        vector2 = vector2.view(B, -1)  # => [B, 104]
        
        # Combine => [B, 208] if vector_dim=104
        vec_input = torch.cat([vector1, vector2], dim=1)  # => [B, 2*vector_dim]
        
        # Pass through fc + norm + relu => [B, 128]
        vec_features = self.vector_fc(vec_input)
        vec_features = self.vector_norm(vec_features)
        vec_features = self.relu(vec_features)
        
        # -----------------------------------------------------------------
        # Process scalars individually
        # -----------------------------------------------------------------
        # If scalars is [B,1,5], squeeze dim=1 => [B,5]
        if scalars.dim() == 3 and scalars.size(1) == 1:
            scalars = scalars.squeeze(1)  # => [B, scalar_count]
        
        # Encode each scalar => 16 dims
        scalar_list = []
        for i, fc_layer in enumerate(self.scalar_fc_list):
            # scalars[:, i] => [B], expand => [B,1]
            scalar_i = scalars[:, i].unsqueeze(1)
            scalar_i_emb = fc_layer(scalar_i)  # => [B,16]
            scalar_list.append(scalar_i_emb)
        
        # Concatenate => [B, scalar_count*16]
        scalar_features = torch.cat(scalar_list, dim=1)  # => e.g. [B, 80] if 5 scalars
        scalar_features = self.scalar_norm(scalar_features)
        scalar_features = self.relu(scalar_features)
        
        # -----------------------------------------------------------------
        # Combine vector features + scalar features => feed to latent_to_image
        # -----------------------------------------------------------------
        combined = torch.cat([vec_features, scalar_features], dim=1)  
        # => [B, 128 + scalar_count*16]
        
        latent_image = self.latent_to_image(combined)     # => [B, 4096] if 64x64
        latent_image = latent_image.view(B, 1, 64, 64)    # => [B,1,64,64]
        
        # -----------------------------------------------------------------
        # UNet encoder
        # -----------------------------------------------------------------
        f1, p1 = self.encoder1(latent_image)  # 64->32
        f2, p2 = self.encoder2(p1)            # 32->16
        f3, p3 = self.encoder3(p2)            # 16->8
        
        # Bottleneck
        btl = self.bottleneck(p3)            # 8->8
        
        # -----------------------------------------------------------------
        # UNet decoder
        # -----------------------------------------------------------------
        u3 = self.decoder3(btl, f3)          # 8->16
        u2 = self.decoder2(u3, f2)           # 16->32
        u1 = self.decoder1(u2, f1)           # 32->64
        
        # Upsample from 64->128
        up1 = self.up1(u1)  # => [B,16,128,128]
        up1 = self.relu(up1)
        
        # Final conv => [B,1,128,128]
        output_image = self.final_conv(up1)
        output_image = self.relu(output_image)
        
        # Resize 128->131
        output_image = self.final_resize(output_image)
        
        return output_image

###############################################################
# embedded UnetDecoder model

class UNetDecoder(nn.Module):
    """
    Takes a 2-channel 131x131 image and:
      1) Two-step downsample to 64x64:
         (a) 131->128 by bilinear up/downsampling
         (b) 128->64 by MaxPool2d
      2) Pass through a UNet (3 encoder levels + bottleneck + 3 decoder levels)
      3) Squeeze channels to 1 (1x64x64)
      4) Flatten and produce reconstructed vectors/scalars
    """
    def __init__(self, vector_dim, scalar_count):
        super(UNetDecoder, self).__init__()
        
        # ReLU for consistency
        self.relu = nn.ReLU(inplace=True)
        
        # --------------------
        # Step 1: 131 -> 128 (non-trainable bilinear interpolation)
        # Step 2: 128 -> 64  (MaxPool2d, standard downsampling)
        # --------------------
        self.downsample_to_128 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.downsample_128_to_64 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --------------------
        # UNet Encoder
        # --------------------
        self.encoder1 = EncoderBlock(in_channels=2,  out_channels=32)   # 64x64 -> 32x32
        self.encoder2 = EncoderBlock(in_channels=32, out_channels=64)   # 32x32 -> 16x16
        self.encoder3 = EncoderBlock(in_channels=64, out_channels=128)  # 16x16 -> 8x8
        
        # Replace bottleneck with the TransformerBottleneck
        self.bottleneck = TransformerBottleneck(
            in_channels=128, 
            mid_channels=256, 
            n_heads=4, 
            dim_feedforward=1024
        )
        
        # --------------------
        # UNet Decoder
        # --------------------
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128) # 8x8  -> 16x16
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)  # 16x16 -> 32x32
        self.decoder1 = DecoderBlock(in_channels=64,  out_channels=32)  # 32x32 -> 64x64
        
        # --------------------
        # Final conv to get single latent channel
        # --------------------
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        # --------------------
        # Flatten + FC layers
        # --------------------
        # Flatten from (1,64,64) => 4096
        self.fc_main = nn.Linear(64 * 64, 512)
        self.fc_norm = nn.LayerNorm(512)
        
        self.vector_fc1 = nn.Linear(512, vector_dim)
        self.vector_fc2 = nn.Linear(512, vector_dim)
        self.scalar_fc  = nn.Linear(512, scalar_count)

    def forward(self, x):
        """
        x: (batch_size, 2, 131, 131) input image
        Returns:
            reconstructed_vector1, reconstructed_vector2, reconstructed_scalars
        """
        # --------------------
        # Two-step downsampling
        # --------------------
        x = self.downsample_to_128(x)       # (B, 2, 128, 128)
        x = self.downsample_128_to_64(x)    # (B, 2, 64, 64)

        # --------------------
        # UNet Encoder
        # --------------------
        f1, p1 = self.encoder1(x)   # (B, 32, 32, 32)
        f2, p2 = self.encoder2(p1)  # (B, 64, 16, 16)
        f3, p3 = self.encoder3(p2)  # (B, 128, 8, 8)
        
        # Bottleneck
        btl = self.bottleneck(p3)   # (B, 256, 8, 8)
        
        # --------------------
        # UNet Decoder
        # --------------------
        u3 = self.decoder3(btl, f3) # (B, 128, 16, 16)
        u2 = self.decoder2(u3, f2)  # (B, 64,  32, 32)
        u1 = self.decoder1(u2, f1)  # (B, 32,  64, 64)
        
        # --------------------
        # Final conv to single channel
        # --------------------
        out = self.final_conv(u1)   # (B, 1, 64, 64)
        out = self.relu(out)        # ReLU activation

        # --------------------
        # Flatten and feed into FC layers
        # --------------------
        out = out.view(out.size(0), -1)  # (B, 4096)
        
        # Main FC
        out = self.fc_main(out)     # (B, 512)
        out = self.fc_norm(out)
        out = self.relu(out)
        
        # Separate heads for reconstructed vectors & scalars
        reconstructed_vector1 = self.vector_fc1(out)
        reconstructed_vector2 = self.vector_fc2(out)
        reconstructed_scalars = self.scalar_fc(out)
        
        return reconstructed_vector1, reconstructed_vector2, reconstructed_scalars