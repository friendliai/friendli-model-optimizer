<!---
Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.
-->

<h2><p align="center">Friendli Model Optimizer (FMO) for Supercharge Generative AI Serving 🚀</p></h2>

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
FMO is a tool that provides model optimizations for efficient generative AI serving with [Friendli Engine](https://friendli.ai/solutions/engine/).
It provides features to improve generative AI serving performance without compromising task accuracy.

FMO is designed to work with Huggingface pretrained model, which can be loaded using ['PreTrainedModel.from_pretrained()'](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).

> [!NOTE]
> The Huggingface model architectures that can be optimized with FMO is specified in [Supported Features & Model Architecture](#supported-features--model-architecture).


# Table of Contents
- [Quick Installation](#quick-installation)
- [Supported Features & Model Architecture](#supported-features--model-architecture)
- [User Guides](#user-guides)
- [How to Serve Optimized Model?](#how-to-serve-optimized-model-with-frinedli-engine)


# Quick Installation
```bash
pip install friendli-model-optimizer
```


# Supported Features & Model Architecture
FMO currently supports the following PTQ(Post-Training Quantization) techniques:

## FP8

FP8 is an 8-bit floating-point format that offers a higher dynamic range than INT8,
making it better suited for quantizing both weights and activations.
This leads to increased throughput and reduced latency while maintaining high output quality with minimal degradation.

> [!IMPORTANT]
> FP8 is only supported by NVIDIA Ada, Hopper, and Blackwell GPU architectures.

> [!NOTE]
> For now, we only support E4M3 (4-bit exponent and 3-bit mantissa) encoding format.

### Supported Model Architecutre for FP8 Quantization
- `LlamaForCausalLM`
- `MistralForcausalLM`
- `CohereForCausalLM`
- `Qwen2ForCausalLM`
- `Gemma2ForCausalLM`
- `Phi3ForCausalLM`
- `MptForCausalLM`
- `ArcticForCausalLM`
- `MixtralForCausalLM`


## INT8

INT8 Quantization represents weights and activations using the INT8 format with acceptable accuracy drops.
Friendli Engine enables dynamic activation scaling, where scales are computed on the fly during runtime.
Thus, FMO only quantize weight, and Friendli Engine will load quantized weight.

### Supported Model Architecutre for INT8 Quantization
- `LlamaForCausalLM`
- `MistralForcausalLM`
- `CohereForCausalLM`
- `Qwen2ForCausalLM`
- `Gemma2ForCausalLM`


# User Guides
You can run the quantization processes with the command below:
```bash
fmo quantize \
--model-name-or-path $MODEL_NAME_OR_PATH \
--output-dir $OUTPUT_DIR \
--mode $QUANTIZATION_SCHEME \
--device $DEVICE \
--offload
```
The command line arguments means :
- **`model-name-or-path`**: Hugging Face pretrained model name or directory path of the saved model checkpoint.
- **`output-dir`**: Directory path to save the quantized checkpoint and related configurations.
- **`mode`**: Qantization techniques to apply. You can use `fp8`, `int8`.
- **`device`**: Device to run the quantization process. Defaults to "cuda:0".
- **`offload`**: When enabled, this option significantly reduces GPU memory usage by offloading model layers onto CPU RAM. Defaults to true.

> [!TIP]
> If you want to use more advanced quantization options(e.g., calibration dataset), Please checkout our [official documentations](https://docs.friendli.ai/guides/container/running_friendli_container/quantization).

> [!NOTE]
> You can use a configuration YAML file

## Example: Run FP8 uantization with Meta-Llama-3-8B-Instruct
```bash
export MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
export OUTPUT_DIR="./"

fmo quantize \
--model-name-or-path $MODEL_NAME_OR_PATH \
--output-dir $OUTPUT_DIR \
--mode "fp8" \
--device "cuda:0" \
--offload
```

# How to serve optimized model with Frinedli Engine?
If your optimized model is ready, now, you can serve the model with Friendli Engine.\
Please checkout our [official documentations](https://docs.friendli.ai/guides/container/running_friendli_container/quantization) to learn more!
