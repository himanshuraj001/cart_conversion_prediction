
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

1. **Environment Setup:** Ensure you have a CUDA-compatible environment. Install the required dependencies:
```bash
pip install torch torchvision numpy pandas pillow tqdm transformers peft sentence-transformers
```
### Hugging Face Token

Export your Hugging Face authentication token to access gated models like Llama 3.2:

```bash
export HF_TOKEN="your_huggingface_token_here"
```


### Data Preparation
The framework is evaluated on the SLAKE (English subset) and VQA-Med 2019 datasets.
Your data directories should be structured as follows:

* Images: Original medical scans (e.g., CT, MRI, X-Ray).
* Masks: Binary anatomical region masks corresponding to the original scans (generated via MedCLIP-SAM).
* Annotations: JSON files containing the multiple-choice question format. Each entry should include img_name, question, options, correct_answer, modality_group, and organ_group.

Update the constant paths (e.g., TRAIN_DATA_JSON, TRAIN_IMAGE_ROOT, TRAIN_MASK_DIR) in train_demo.py and test_demo.py to point to your local dataset locations.

### Training (MAFt)
To finetune the base VLM on the source domain using LoRA and Mask-Aware Finetuning:

```bash
python train_demo.py
```
Training Details:
* Utilizes 8-bit AdamW optimizer and gradient checkpointing to save VRAM.
* Applies a custom masking strategy in the collate function so the model calculates loss exclusively on the generated medical answer, not the prompt.

### Inference (AAIN + WMMER)
To run the evaluation pipeline on the target domain:

```bash
python test_demo.py
```
Inference Details:
* Dynamically normalizes target images using cached source domain statistics (AAIN).
* Uses clip-ViT-B-32 to retrieve the best 1-shot example from the support set based on a weighted blend of visual and textual similarity (WMMER).

  Update directory paths here too...
  






