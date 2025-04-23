import torch

def LR(img_noisy, threshold=0.99):
    img_noisy = torch.squeeze(img_noisy)
    U, S, Vh = torch.linalg.svd(img_noisy, full_matrices=False)
    energy_total = torch.sum(S**2)
    energy_cumulative = torch.cumsum(S**2, dim=0)
    energy_threshold = threshold * energy_total
    k = torch.searchsorted(energy_cumulative, energy_threshold).item()+1
    U_k = U[:, :k]
    S_k = torch.diag(S[:k])
    Vh_k = Vh[:k, :]
    img_lr = U_k @ S_k @ Vh_k
    img_lr = img_lr.clip(0, 255)
    return img_lr

def im2row(im, img_size, block_size):
    H, W = img_size
    block_h, block_w = block_size

    patch_h = H - block_h + 1
    patch_w = W - block_w + 1
    patch_num = patch_h * patch_w
    block_length = block_h * block_w
    result = torch.zeros((block_length, patch_num))

    patch_idx = 0
    for y in range(patch_w):
        for x in range(patch_h):
            patch = im[x:x+block_h, y:y+block_w]
            result[:, patch_idx] = patch.reshape(-1)
            patch_idx += 1
    return result


def row2im(patches, img_size, block_size):
    H, W = img_size
    block_h, block_w = block_size
    result = torch.zeros((H, W))
    weight = torch.zeros((H, W))

    patch_idx = 0
    for y in range(W-block_w+1):
        for x in range(H-block_h+1):
            patch = patches[:, patch_idx]
            result[x:x+block_h, y:y+block_w] += patch.reshape(8, 8)
            weight[x:x+block_h, y:y+block_w] += 1
            patch_idx += 1
    
    # weight = torch.clamp(weight, min=1e-6)
    result /= weight
    return result

def LLR(img_noisy, k, block_size=[8, 8]):
    img_noisy = torch.squeeze(img_noisy)
    img_size = img_noisy.shape
    patches = im2row(img_noisy, img_size, block_size)
    U, S, Vh = torch.linalg.svd(patches, full_matrices=False)
    U_k = U[:, :k]
    S_k = torch.diag(S[:k])
    Vh_k = Vh[:k, :]
    patch_lr = U_k @ S_k @ Vh_k
    img_llr = row2im(patch_lr, img_size, block_size)
    img_llr = img_llr.clip(0, 255)
    return img_llr

def frequency_refinement(Z, sigma):
        # print(sigma)
        # Vectorized frequency grid computation
        k1, k2 = Z.size(2), Z.size(3)
        i = torch.arange(k1, device=Z.device).float()
        j = torch.arange(k2, device=Z.device).float()
        freq_grid = torch.sqrt(i[:, None]**2 + j[None, :]**2) / torch.sqrt(torch.tensor(k1**2 + k2**2))
        

        # Normalize to [0, 1]
        Zmin=Z.min()
        Zmax=Z.max()
        Z = (Z - Zmin) / (Zmax - Zmin + 1e-8)

        # Compute Wiener-like weights (attenuate low-SNR frequencies)
        Z_dct = torch.fft.fft2(Z, dim=(2, 3))
        # print(Z.shape)
        power_spectrum = torch.abs(Z_dct)**2 /(Z_dct.shape[2]*Z_dct.shape[3])
        noise_power = ((sigma*2)**2) * (k1 * k2)  # Noise variance scales with patch size
        # print(torch.mean(power_spectrum),torch.min(power_spectrum),torch.max(power_spectrum))

        # noise_power = self.noise_estimate(Z_dct)
        wiener_weights = power_spectrum / (power_spectrum + noise_power)
                
        # Apply filtering
        Z_dct_filtered = Z_dct * wiener_weights
        Z_freq_filtered = torch.fft.ifft2(Z_dct_filtered, dim=(2, 3)).real
        
        # Denormalize from [0, 1]
        Z = Z_freq_filtered * (Zmax - Zmin) + Zmin

        # Adaptive blending (TODO reduce alpha for low-noise patches)
        alpha = 0.7  # Fixed for now
        return Z#(1 - alpha) * Z + alpha * Z_freq_filtered