from lichi import LIChI
import torch
import torch.nn as nn
import torch.nn.functional as F

class LIChI3D(LIChI):
    def __init__(self, temp_sigma=1.0):
        super().__init__()
        self.temp_sigma = temp_sigma  # Controls temporal weighting
        self.temporal_radius = 2  # Default temporal search radius
        
    def forward(self, input_data, sigma=25.0, p3d=(3,8,8), k3d=24,
               temp_radius=2, M=6):
        """Process input [N, C, T, H, W]"""
        # Store temporal parameters
        self.temporal_radius = temp_radius
        self.p3d = p3d
        self.k3d = k3d
        
        # Initialize parameters for the parent class
        self.set_parameters(sigma=sigma, p1=p3d[1],p2=p3d[2], k1=k3d, M=M)
        
        # Convert input to [N, C, T, H, W] format
        z = input_data.permute(0, 1, 4, 2, 3)  # [N,C,H,W,T]->[N,C,T,H,W]
        x = self.step1(z, sigma)
        
        # Iterative refinement
        for m in range(M):
            tau = (1 - (m+1)/M) * 0.75
            x, z, indices = self.step2(z, x, input_data, sigma, tau)
            
        return z.permute(0, 1, 3, 4, 2)  # [N,C,T,H,W]->[N,C,H,W,T]

    def block_matching(self, input_x, k, p, w, s, temporal_radius=2):
        """3D version with temporal search window"""
        
        def block_matching_3d_aux(input_x_pad, k, p, v, s, temp_rad=temporal_radius):
            N, C, T_pad, H_pad, W_pad = input_x_pad.shape
            
            # Calculate patch counts after unfolding
            T_patches = T_pad - p[0] + 1
            H_patches = (H_pad - p[1]) // s + 1
            W_patches = (W_pad - p[2]) // s + 1
            
            # Generate valid reference positions within patch dimensions
            valid_t = torch.arange(temp_rad, T_patches - temp_rad, device=input_x_pad.device)
            valid_h = torch.arange(v, (H_pad - p[1] - v) // s + 1, device=input_x_pad.device)
            valid_w = torch.arange(v, (W_pad - p[2] - v) // s + 1, device=input_x_pad.device)
        
            # Create grid of valid indices
            t, h, w = torch.meshgrid(valid_t, valid_h, valid_w, indexing='ij')
            
            # Convert to linear indices using patch counts
            ref_indices = (
                t * (H_patches * W_patches) +
                h * W_patches + 
                w
            ).flatten()
        
            # Extract all patches and validate indices
            patches = input_x_pad.unfold(2, p[0], 1).unfold(3, p[1], s).unfold(4, p[2], s)
            patches = patches.contiguous().view(N, C, -1, p[0], p[1], p[2])
            
            # Ensure indices are within valid range
            max_index = patches.size(2) - 1
            ref_indices = torch.clamp(ref_indices, 0, max_index)
        
            # Compute distances using validated indices
            dist_matrix = torch.cdist(
                patches.flatten(3), 
                patches[:, :, ref_indices].flatten(3)
            )
            
            # Get top-k NEAREST neighbors
            _, indices = torch.topk(dist_matrix, k, dim=2, largest=False)
            
            return ref_indices[indices] # Return original patch indices
            
        assert len(p) == 3,  # Ensure p is (temporal, height, width)
        v = w // 2
        input_x_pad = F.pad(input_x, [v]*4 + [temporal_radius]*2, mode='reflect')
        return block_matching_3d_aux(input_x_pad, k, p, v, s)
        
    def gather_groups(self, input_y, indices, k, p):
        N, C, T, H, W = input_y.shape
        patches = input_y.unfold(2, p[0], 1).unfold(3, p[1], 1).unfold(4, p[2], 1)
        patches = patches.contiguous().view(N, C, -1, p[0], p[1], p[2])
        
        # Clamp indices to valid range
        max_idx = patches.size(2) - 1
        indices = torch.clamp(indices, 0, max_idx)
        
        # Expand indices to match patch dimensions
        indices = indices.view(N, 1, -1, k, 1, 1, 1).expand(-1, C, -1, -1, -1, -1, -1)
        
        return torch.gather(patches.unsqueeze(3), dim=2, index=indices).squeeze(3)
    
    def aggregate(self, X_hat, weights, indices, original_shape, p):
        """Reconstruct from 3D patches"""
        N, C, T, H, W = original_shape
        output = torch.zeros(N, C, T, H, W, device=X_hat.device)
        count = torch.zeros_like(output)
        
        for t in range(T - p[0] + 1):
            for h in range(H - p[1] + 1):
                for w in range(W - p[2] + 1):
                    output[..., t:t+p[0], h:h+p[1], w:w+p[2]] += X_hat[..., t, h, w, :] * weights
                    count[..., t:t+p[0], h:h+p[1], w:w+p[2]] += weights
                    
        return output / count.clamp(min=1e-6)
        
    def denoise2(self, Z, X, Y, sigma, tau, temporal_weights=True):
        """Add temporal weighting during aggregation"""
        # Original denoising logic
        X_hat, Z_hat, weights = super().denoise2(Z, X, Y, sigma, tau)
        
        if temporal_weights:
            # Calculate temporal similarity weights
            t_dist = torch.arange(Z.shape[2], device=Z.device).float()
            t_weights = torch.exp(-(t_dist[None,:] - t_dist[:,None])**2 / (2*self.temp_sigma**2))
            weights = weights * t_weights.view(1,1,-1,1,1)
            
        return X_hat, Z_hat, weights

    def step1(self, input_y, sigma):
        """
        Modified for inputs [N, C, T, H, W]
        """
        
        # Get dimensions including temporal axis
        N, C, T, H, W = input_y.size() 
        k, p, w, s = self.k1, self.p1, self.w, self.s
        
        # Average over color channels (keep temporal dimension)
        y_mean = torch.mean(input_y, dim=1, keepdim=True)  # [N,1,T,H,W]
        
        # Block matching needs to handle temporal dimension
        indices = self.block_matching(y_mean, k, (1, p, p), w, s, temporal_radius=2)  # Temporal patch size=1 for first step
        
        # Gather 3D patches
        Y = self.gather_groups(input_y, indices, k, (1, p, p))  # [N, B, k, C, 1, p, p]
        
        # Denoise
        X_hat, weights = self.denoise1(Y, sigma)
        
        # Aggregate
        x_hat = self.aggregate(X_hat, weights, indices, (T, H, W), (1, p, p))
        
        return x_hat

    def step2(self, input_z, input_x, input_y, sigma, tau, indices=None):
        """
        Modified for inputs [N, C, T, H, W]
        """
  
        N, C, T, H, W = input_y.size()
        k, p, w, s = self.k2, self.p2, self.w, self.s
        z_block = torch.mean(input_z, dim=1, keepdim=True)  # [N,1,T,H,W]
        
        # Block matching with temporal radius
        if indices is None:
            indices = self.block_matching(z_block, k, p, w, s, self.temporal_radius)
        
        # Gather 3D patches
        X = self.gather_groups(input_x, indices, k, p)
        Y = self.gather_groups(input_y, indices, k, p)
        Z = self.gather_groups(input_z, indices, k, p)
        
        # Denoise with temporal weighting
        X_hat, Z_hat, weights = self.denoise2(Z, X, Y, sigma, tau, temporal_weights=True)
        
        # Aggregate 3D patches
        x_hat = self.aggregate(X_hat, weights, indices, (T, H, W), p)
        z_hat = self.aggregate(Z_hat, weights, indices, (T, H, W), p)
        
        return x_hat, z_hat, indices