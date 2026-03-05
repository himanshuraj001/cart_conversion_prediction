
# Leveraging Spatial Statistics for Domain Adaptation of VLMs in Medical VQA

Official PyTorch implementation of the paper **"Leveraging Spatial Statistics for Domain Adaptation of VLMs in Medical VQA"**. 

This repository provides the code for the **Spatial Semantics Aware Domain Adaptation (SSADA)** framework. SSADA is designed to tackle domain generalization challenges in Medical Visual Question Answering (VQA) by addressing the complex spatial semantics and pronounced distributional shifts inherent in medical imaging.

## Framework Overview

The SSADA framework integrates finetuning and prompt-based in-context learning through three core modules:

1. **Mask-Aware Finetuning (MAFt):** Incorporates anatomical region-of-interest masks during the finetuning of Large Vision-Language Models (LVLMs). This promotes spatially grounded semantic learning, preventing the model from overfitting to domain-specific scanner noise.
2. **Anatomy-Aware Instance Normalization (AAIN):** Replaces standard global normalization with region-conditioned instance normalization. It leverages anatomical priors to compute statistics for foreground and background regions independently, reducing intensity discrepancies across domains.
3. **Weighted Multi-Modal Example Retrieval (WMMER):** Dynamically constructs few-shot prompts by retrieving exemplars from a target domain support set. It jointly optimizes for both anatomical (visual) and semantic (textual) relevance.

## Supported Models

The code is natively compatible with the Hugging Face `transformers` library and has been validated on:
* `meta-llama/Llama-3.2-11B-Vision-Instruct` (Default in provided scripts)
* `Qwen/Qwen2-VL-7B-Instruct` & `2B-Instruct`
* `google/gemma-3-12b-it`
* `HuggingFaceTB/SmolVLM-Instruct`

## Installation and Setup

1. **Environment Setup:** Ensure you have a CUDA-compatible environment (e.g., NVIDIA A6000 or similar for 11B parameter models). Install the required dependencies:

```bash
pip install torch torchvision numpy pandas pillow tqdm transformers peft sentence-transformers
