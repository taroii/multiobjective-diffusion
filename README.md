# Multiobjective Diffusion

This repository is the official implementation of **paper title**. 

We propose...

This repository contains...

## Requirements

Create a fresh environment and install the Python dependencies:

```bash
conda create -n multiobjective-diffusion python=3.11
conda activate multiobjective-diffusion
pip install -r requirements.txt
```

Install PyTorch separately to match your hardware. On CPU only:

```bash
pip install torch torchvision
```

On CUDA (example — substitute your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

See https://pytorch.org/get-started/locally/ for the correct command for your system.

## System Specifications of Reported Runs

For all of the results we report in our paper, we use a dedicated Linux server with the following specifications:

  

| **Component** | **Details** |

|--------------|-------------|

| **Machine** | Lenovo ThinkStation P3 Tower Gen 2 |

| **CPU** | Intel Core Ultra 9 285K (24 cores, up to 7.3 GHz) |

| **GPU** | NVIDIA RTX 4000 Ada Generation (Lovelace) |

| **RAM** | 48 GB |

| **Storage** | 1.86 TB NVMe SSD + 2×10.9 TB HDD (≈ 23.7 TB total) |

| **OS** | Ubuntu 24.04.2 LTS (Noble Numbat) |

| **Driver** | NVIDIA 580.126.09 |