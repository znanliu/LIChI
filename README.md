# Linear Combinations of Patches Are Unreasonably Effective for Single-Image Denoising (IEEE TIP'24)
Project team:  Wenxin Hu, Yaoxin Li, Zhongnan Liu, Taewon Yang

Fall25 EECS 556

Extending on the work of Sébastien Herbreteau and Charles Kervrann

## Project Organization
### Extensions:
### 1.  Comparing the performance with LR, LLR and DIP methods
- [denoising_methods.py](./denoising_methods.py)

- [U-net.ipynb](./U-net.ipynb)
### 2.  Overcoming LIChI limitations
- [extension_tw.ipynb](./extension_tw.ipynb)
### 3.  Patch-based Denoising in Frequency Domain
- [frequency_hybrid_lichi_selina.ipynb](./frequency_hybrid_lichi_selina.ipynb): README, code, and plots. Requirements specified within the code.
### 4.  Application on 3D Medical Data
- [lichi3d.py](./lichi3d.py)

## Requirements

Here is the list of libraries you need to install to execute the code:
* Python 3.8
* Pytorch 2.2
* Torchvision 0.17
* Einops 0.7.0

## Install

To install in an environment using pip:

```
python -m venv .lichi_env
source .lichi_env/bin/activate
pip install /path/to/LIChI
```

## Demo

To denoise an image with LIChI (remove ``--add_noise`` if it is already noisy):
```
python ./demo.py --sigma 15 --add_noise --in ./test_images/cameraman.png --out ./denoised.png
```

Or use directly the Pytorch class LIChI within your code:
```
m_lichi = LIChI() # instantiate the LIChI class
y = 15 * torch.randn(1, 1, 100, 100) # image of pure Gaussian noise with variance 15^2
x_hat = m_lichi(y, sigma=15, constraints='affine', method='n2n', p1=11, p2=6, k1=16, k2=64, w=65, s=3, M=9)
```
(see the meaning of the parameters in file lichi.py, method set_parameters)

## Results

### Gray denoising
The PSNR (dB) results of different methods on three datasets corrupted with synthetic white Gaussian noise
and sigma = 5, 15, 25, 35 and 50. Best among each category (unsupervised or supervised) is in bold. Best among each
subcategory is underlined

<img width="1066" alt="results_psnr" src="https://user-images.githubusercontent.com/88136310/205091125-6dbbf47c-d639-4485-8a95-f649ccc44efa.png">


### Complexity
We want to emphasize that  LIChI is relatively fast. We report here the execution times of different algorithms. It is
provided for information purposes only, as the implementation, the language used and the machine on which the code is run, highly influence the  results. The CPU used is a 2,3 GHz Intel Core i7 and the GPU is a GeForce RTX 2080 Ti. LIChI has been entirely written in Python with Pytorch so it can run on GPU unlike its traditional counterparts. 


Running time (in seconds) of different methods on images of size 256x256. Run times are given on CPU and GPU if available.

<img width="285" alt="results_running_time" src="https://user-images.githubusercontent.com/88136310/205092027-11aa0770-17fd-40c1-b9d7-1973c56732b3.png">


## Acknowledgements

This work was supported by Bpifrance agency (funding) through the LiChIE contract. Computations  were performed on the Inria Rennes computing grid facilities partly funded by France-BioImaging infrastructure (French National Research Agency - ANR-10-INBS-04-07, “Investments for the future”).

## Citation
```BibTex
@ARTICLE{10639330,
  author={Herbreteau, Sébastien and Kervrann, Charles},
  journal={IEEE Transactions on Image Processing}, 
  title={Linear Combinations of Patches are Unreasonably Effective for Single-Image Denoising}, 
  year={2024},
  volume={33},
  number={},
  pages={4600-4613},
  doi={10.1109/TIP.2024.3436651}}
```
