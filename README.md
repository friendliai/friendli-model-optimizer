<!---
Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.
-->

<h2><p align="center">Friendli Model Optimizer (FMO) for supercharging generative AI serving 🚀</p></h2>

<p align="center">
  <a href="https://github.com/friendliai/friendli-model-optimizer/actions/workflows/ci.yaml">
    <img alt="CI Status" src="https://github.com/friendliai/friendli-model-optimizer/actions/workflows/ci.yaml/badge.svg">
  </a>
  <a href="https://pypi.org/project/friendli-model-optimizer">
    <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/friendli-model-optimizer?logo=Python&logoColor=white">
  </a>
  <a href="https://pypi.org/project/friendli-model-optimizer/">
      <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/friendli-model-optimizer?logo=PyPI&logoColor=white">
  </a>
  <a href="https://docs.friendli.ai/">
    <img alt="Documentation" src="https://img.shields.io/badge/read-doc-blue?logo=ReadMe&logoColor=white">
  </a>
  <a href="https://github.com/friendliai/friendli-model-optimizer/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=Apache">
  </a>
</p>


# Overview
Friendli Model Optimizer (FMO) is a tool that provides model optimizations for efficient generative AI serving with [Friendli Engine](https://friendli.ai/solutions/engine/).
The optimizations improve generative AI serving performance without compromising task accuracy.

FMO is designed to work with [`transformers`](https://huggingface.co/docs/transformers/index) library. You can optimize the model in Hugging Face Model Hub using FMO.

FMO currently supports PTQ(Post Training Quantization) algorithms, FP8, INT8 and AWQ.

[!NOTE]
FMO currently utilizes **a single GPU** for running optimizations. But, it can generate optimized model checkpoints for large models like LLaMA-3.1-70B and LLaMA-3.1-405B! Additionally, even for FP8 precision, you are not restricted to using GPUs that support FP8.


# What's NEW? (latest: v0.10.0)
- Bug Fix for FP8 pedantic level 2
- Support Python3.12, Python3.13.


# Table of Contents
- [Quick Installation](#quick-installation)
- [Supported Features & Model Architecture](#supported-features--model-architecture)
- [User Guides](#user-guides)
  - [Serving an Optimized Model](#how-to-serve-an-optimized-model-with-friendli-engine)
  - [How to improve an optimized model quality with calibration dataset?](#how-to-improve-an-optimized-model-quality-with-calibration-dataset)

# Quick Installation
```bash
pip install friendli-model-optimizer
```

# Supported Features & Model Architecture
FMO currently supports the following PTQ (Post-Training Quantization) techniques:

## FP8

FP8 is an 8-bit floating-point format that offers a higher dynamic range than INT8,
making it better suited for quantizing both weights and activations.
This leads to increased throughput and reduced latency while maintaining high output quality with minimal degradation.

FMO offers a pedantic level setting, which controls the trade-off between accuracy and processing time for FP8.
Higher pedantic levels provide more accurate model but can increase the time required to generate quantized models, and may sometimes slow down inference. Lower pedantic levels allow for faster quantization, though they may reduce model accuracy. Each quantization mode supports different ranges of pedantic levels.

FP8 support 1-2 pedantic level. Defaults to 1.

> [!IMPORTANT]
> FP8 is only supported by NVIDIA Ada, Hopper, and Blackwell GPU architectures.

> [!NOTE]
> For now, we only support the E4M3 (4-bit exponent and 3-bit mantissa) encoding format.

### Supported Model Architectures for FP8 Quantization
- `CohereForCausalLM`
- `ExaoneForCausalLM`
- `Gemma2ForCausalLM`
- `LlamaForCausalLM`
- `MistralForcausalLM`
- `MixtralForCausalLM`
- `MptForCausalLM`
- `Phi3ForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen2VLForConditionalGeneration`

## INT8

INT8 Quantization represents weights and activations using the INT8 format with acceptable accuracy drops.
Friendli Engine enables dynamic activation scaling, where scales are computed on the fly during runtime.

### Supported Model Architectures for INT8 Quantization
- `CohereForCausalLM`
- `ExaoneForCausalLM`
- `Gemma2ForCausalLM`
- `LlamaForCausalLM`
- `MistralForcausalLM`
- `MixtralForCausalLM`
- `MptForCausalLM`
- `Phi3ForCausalLM`
- `Qwen2ForCausalLM`


## AWQ
Activation-Aware Weight Quantization (AWQ) is a technique that optimizes neural networks for efficiency without compromising accuracy. Unlike traditional weight quantization methods, AWQ leverages a deep understanding of the data distribution within neural networks during inference.

To learn more about AWQ, refer to [this article](https://friendli.ai/blog/activation-aware-weight-quantization-llm).

### Supported Model Architectures for INT8 Quantization
- `CohereForCausalLM`
- `ExaoneForCausalLM`
- `Gemma2ForCausalLM`
- `LlamaForCausalLM`
- `MistralForcausalLM`
- `MixtralForCausalLM`
- `MptForCausalLM`
- `Phi3ForCausalLM`
- `Qwen2ForCausalLM`

# User Guides

You can run the quantization processes with the command below:
```bash
fmo quantize \
--model-name-or-path $MODEL_NAME_OR_PATH \
--output-dir $OUTPUT_DIR \
--mode $QUANTIZATION_SCHEME \
--pedantic-level $PEDANTIC_LEVEL
--device $DEVICE \
--offload
```
The command line arguments means :
- **`model-name-or-path`**: Hugging Face pretrained model name or directory path of the saved model checkpoint.
- **`output-dir`**: Directory path to save the quantized checkpoint and related configurations.
- **`mode`**: Quantization techniques to apply. You can use `fp8`, `int8`.
- **`pedantic-level`**: Represent to accuracy-latency trade-off. Higher pedantic level ensure a more accurate representaition of the model, but increase the quantization processing time. Defaults to 1.
- **`device`**: Device to run the quantization process. Defaults to "cuda".
- **`offload`**: When enabled, this option significantly reduces GPU memory usage by offloading model layers onto CPU RAM. Defaults to False.

## Example: Run FP8 quantization with Meta-Llama-3-8B-Instruct
```bash
export MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
export OUTPUT_DIR="./"
export CUDA_VISIBLE_DEVICES=0

fmo quantize \
--model-name-or-path $MODEL_NAME_OR_PATH \
--output-dir $OUTPUT_DIR \
--mode "fp8" \
--device "cuda" \
```

Once your optimized model is ready, you can serve the model with Friendli Engine.
Please check out our [official documentation](https://docs.friendli.ai/guides/container/serving_quantized_models) to learn more!

## How to improve an optimized model quality with calibration dataset?
Using a calibration dataset that closely resembles the data to be generated during deployment can improve the quality of the quantized model when deployed.

Currently, we use the default calibration dataset with the following specifications, which serve as a great starting point for calibration:
- **Dataset**: [`cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail) (version 3.0.0)
- **Split Name of Dataset**: test
- **Column Name of Dataset**: article
- **Number of samples**: 512
- **Sequence length**: 1024

These settings offer a solid foundation. However, further tuning may be necessary based on your specific needs.\
For instance, consider using a custom dataset in the following scenarios:

 * If the generated text is primarily in a language other than English, while optimization results may still be acceptable, including texts in the primary language is a good practice.

 * If the generated texts are highly structured (e.g., JSON, XML) rather than plain text, using a custom dataset that better matches this structure can lead to improved performance.

[!TIP]
> If the optimized model continues to experience significant accuracy drops, you may try increasing the sample size or extending the sequence length to enhance performance.


# Support & Issues
If you have any questions or issues, please feel free to [open an issue](https://github.com/friendliai/friendli-model-optimizer/issues/new) in this repository.
