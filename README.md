# 🚀 CUDA-Accelerated McEliece KEM (FYP2)

![GitHub repo size](https://img.shields.io/github/repo-size/DesmondJS/GPU_McEliece)
![License](https://img.shields.io/badge/License-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-12.4-blue)
![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-orange)

## 📖 Overview
This repository contains my **Final Year Project 2 (FYP2)** at Universiti Tunku Abdul Rahman (UTAR), focusing on **GPU parallelization** of the [Classic McEliece](https://classic.mceliece.org/) post-quantum cryptosystem using **CUDA**.

The goal was to accelerate **encryption, decryption, encapsulation, and decapsulation** by offloading computationally intensive tasks (syndrome generation, error vector formation, FFT, Benes network, etc.) to NVIDIA GPUs.  
Performance was benchmarked and compared with an optimised CPU vectorized implementation, showing **significant throughput improvements**.

## ✨ Features
- ✅ **Full KEM flow:** Encapsulation, Decapsulation, Encryption, Decryption  
- ✅ **CUDA kernels** for FFT, Benes network, error vector generation  
- ✅ **Benchmarking scripts** – CPU vs GPU, multiple `num_blocks` configurations  
- ✅ **CSV output** for results & plotting  
- ✅ **Full FYP2 Report** (PDF) included  

## 🖥️ Tested Environment
| Component | Specification |
|----------|---------------|
| OS | Ubuntu 24.04.2 LTS |
| GPU | NVIDIA GeForce GTX 1650 (4 GB) |
| CUDA Toolkit | 12.4 |
| CPU | AMD Ryzen 5 3550H @ 2.10 GHz |
| RAM | 12 GB |
| Driver Version | 550.144.03 |

## 📊 Key Results (Throughput in Bytes/Second)
Here’s a highlight of the performance improvement:

| Num Blocks | CPU Encrypt (B/s) | GPU Encrypt (B/s) | CPU Decrypt (B/s) | GPU Decrypt (B/s) |
|-----------|-----------------|-----------------|-----------------|-----------------|
| 1 | 10,917,018 | 2,314,525 | 2,515,560 | 553,932 |
| 8 | 12,806,684 | 20,305,654 | 2,488,659 | 4,657,336 |
| 32 | 12,219,651 | **81,403,555** | 2,508,264 | **18,515,560** |

📈 **Observation:**  
- CPU throughput stays almost constant — no parallelism (loops run sequentially).  
- GPU throughput rises almost linearly with `num_blocks`, showing strong scaling thanks to parallel execution.
- At 32 blocks, GPU achieves **6.7× faster encryption** and **7.4× faster decryption** compared to CPU.

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repo
- git clone [https://github.com/DesmondJS/GPU_McEliece.git]
- cd GPU_McEliece

### 2️⃣ Build
- make clean
- make

### 3️⃣ Run
- ./run_test

## 🙌 Acknowledgements
This project was completed as part of UTAR FYP under the supervision of Dr. Lee Wai Kong.

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

