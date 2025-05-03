# Models

## Generative Large Language Models

### microsoft/Phi-4-reasoning-plus

https://huggingface.co/microsoft/Phi-4-reasoning-plus

| **Developers** | Microsoft Research |
| --- | --- |
| **Description** | Phi-4-reasoning-plus is a state-of-the-art open-weight reasoning model finetuned from Phi-4 using supervised fine-tuning on a dataset of chain-of-thought traces and reinforcement learning. The supervised fine-tuning dataset includes a blend of synthetic prompts and high-quality filtered data from public domain websites, focused on math, science, and coding skills as well as alignment data for safety and Responsible AI. The goal of this approach was to ensure that small capable models were trained with data focused on high quality and advanced reasoning. Phi-4-reasoning-plus has been trained additionally with Reinforcement Learning, hence, it has higher accuracy but generates on average 50% more tokens, thus having higher latency. |
| **Architecture** | Base model same as previously released Phi-4, 14B parameters, dense decoder-only Transformer model |
| **Inputs** | Text, best suited for prompts in the chat format |
| **Context length** | 32k tokens |

At the high-level overview of the model quality on representative benchmarks. For the tables below, higher numbers indicate better performance:

### Qwen/Qwen3-32B

https://huggingface.co/Qwen/Qwen3-32B

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support, with the following key features:

- **Uniquely support of seamless switching between thinking mode** (for complex logical reasoning, math, and coding) and **non-thinking mode** (for efficient, general-purpose dialogue) **within single model**, ensuring optimal performance across various scenarios.
- **Significantly enhancement in its reasoning capabilities**, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
- **Superior human preference alignment**, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
- **Expertise in agent capabilities**, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
- **Support of 100+ languages and dialects** with strong capabilities for **multilingual instruction following** and **translation**.

### Qwen/Qwen2.5-Omni-7B

https://huggingface.co/Qwen/Qwen2.5-Omni-7B

Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner.

### microsoft/MAI-DS-R1

https://huggingface.co/microsoft/MAI-DS-R1

MAI-DS-R1 is a DeepSeek-R1 reasoning model that has been post-trained by the Microsoft AI team to improve its responsiveness on blocked topics and its risk profile, while maintaining its reasoning capabilities and competitive performance.

FREE DEMO: https://openrouter.ai/microsoft/mai-ds-r1:free

### THUDM/GLM-4-32B-0414

https://huggingface.co/THUDM/GLM-4-32B-0414

The GLM family welcomes new members, the GLM-4-32B-0414 series models, featuring 32 billion parameters. Its performance is comparable to OpenAI’s GPT series and DeepSeek’s V3/R1 series. It also supports very user-friendly local deployment features. GLM-4-32B-Base-0414 was pre-trained on 15T of high-quality data, including substantial reasoning-type synthetic data. This lays the foundation for subsequent reinforcement learning extensions. In the post-training stage, we employed human preference alignment for dialogue scenarios. Additionally, using techniques like rejection sampling and reinforcement learning, we enhanced the model’s performance in instruction following, engineering code, and function calling, thus strengthening the atomic capabilities required for agent tasks. GLM-4-32B-0414 achieves good results in engineering code, Artifact generation, function calling, search-based Q&A, and report generation. In particular, on several benchmarks, such as code generation or specific Q&A tasks, GLM-4-32B-Base-0414 achieves comparable performance with those larger models like GPT-4o and DeepSeek-V3-0324 (671B).

### microsoft/bitnet-b1.58-2B-4T

https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

This repository contains the weights for BitNet b1.58 2B4T, the first open-source, native 1-bit Large Language Model (LLM) at the 2-billion parameter scale, developed by Microsoft Research.

Trained on a corpus of 4 trillion tokens, this model demonstrates that native 1-bit LLMs can achieve performance comparable to leading open-weight, full-precision models of similar size, while offering substantial advantages in computational efficiency (memory, energy, latency).

### meta-llama/Llama-4-Scout-17B-16E-Instruct

https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct

The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding.

These Llama 4 models mark the beginning of a new era for the Llama ecosystem. We are launching two efficient models in the Llama 4 series, Llama 4 Scout, a 17 billion parameter model with 16 experts, and Llama 4 Maverick, a 17 billion parameter model with 128 experts.

### microsoft/Phi-4-multimodal-instruct

https://huggingface.co/microsoft/Phi-4-multimodal-instruct

Phi-4-multimodal-instruct is a lightweight open multimodal foundation model that leverages the language, vision, and speech research and datasets used for Phi-3.5 and 4.0 models. The model processes text, image, and audio inputs, generating text outputs, and comes with 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning, direct preference optimization and RLHF (Reinforcement Learning from Human Feedback) to support precise instruction adherence and safety measures. The languages that each modal supports are the following:

- Text: Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian
- Vision: English
- Audio: English, Chinese, German, French, Italian, Japanese, Spanish, Portuguese

### ds4sd/SmolDocling-256M-preview

https://huggingface.co/ds4sd/SmolDocling-256M-preview

SmolDocling is a multimodal Image-Text-to-Text model designed for efficient document conversion. It retains Docling's most popular features while ensuring full compatibility with Docling through seamless support for DoclingDocuments.

### deepseek-ai/DeepSeek-V3-0324

https://huggingface.co/deepseek-ai/DeepSeek-V3-0324

DeepSeek-V3-0324 demonstrates notable improvements over its predecessor, DeepSeek-V3, in several key aspects.
Significant improvements in benchmark performance:
MMLU-Pro: 75.9 → 81.2 (+5.3)
GPQA: 59.1 → 68.4 (+9.3)
AIME: 39.6 → 59.4 (+19.8)
LiveCodeBench: 39.2 → 49.2 (+10.0)

### mistralai/Mistral-Small-3.1-24B-Instruct-2503

https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503

Building upon Mistral Small 3 (2501), Mistral Small 3.1 (2503) adds state-of-the-art vision understanding and enhances long context capabilities up to 128k tokens without compromising text performance. With 24 billion parameters, this model achieves top-tier capabilities in both text and vision tasks.
This model is an instruction-finetuned version of: Mistral-Small-3.1-24B-Base-2503.

Mistral Small 3.1 can be deployed locally and is exceptionally "knowledge-dense," fitting within a single RTX 4090 or a 32GB RAM MacBook once quantized.

| Model | MMLU | MMLU Pro (5-shot CoT) | MATH | GPQA Main (5-shot CoT) | GPQA Diamond (5-shot CoT ) | MBPP | HumanEval | SimpleQA (TotalAcc) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Small 3.1 24B Instruct** | 80.62% | 66.76% | 69.30% | **44.42%** | **45.96%** | 74.71% | **88.41%** | **10.43%** |
| Gemma 3 27B IT | 76.90% | **67.50%** | **89.00%** | 36.83% | 42.40% | 74.40% | 87.80% | 10.00% |
| GPT4o Mini | **82.00%** | 61.70% | 70.20% | 40.20% | 39.39% | 84.82% | 87.20% | 9.50% |
| Claude 3.5 Haiku | 77.60% | 65.00% | 69.20% | 37.05% | 41.60% | **85.60%** | 88.10% | 8.02% |
| Cohere Aya-Vision 32B | 72.14% | 47.16% | 41.98% | 34.38% | 33.84% | 70.43% | 62.20% | 7.65% |

### CohereForAI/c4ai-command-a-03-2025

https://huggingface.co/CohereForAI/c4ai-command-a-03-2025

C4AI Command A is an open weights research release of a 111 billion parameter model optimized for demanding enterprises that require fast, secure, and high-quality AI. Compared to other leading proprietary and open-weights models Command A delivers maximum performance with minimum hardware costs, excelling on business-critical agentic and multilingual tasks while‬ being deployable on just two GPUs.

### google/gemma-3-27b-pt

https://huggingface.co/google/gemma-3-27b-pt

Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. Gemma 3 models are multimodal, handling text and image input and generating text output, with open weights for both pre-trained variants and instruction-tuned variants. Gemma 3 has a large, 128K context window, multilingual support in over 140 languages, and is available in more sizes than previous versions. Gemma 3 models are well-suited for a variety of text generation and image understanding tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as laptops, desktops or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

| Benchmark | Metric | Gemma 3 PT 1B | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
| --- | --- | --- | --- | --- | --- |
| [**HellaSwag**](https://arxiv.org/abs/1905.07830) | 10-shot | 62.3 | 77.2 | 84.2 | 85.6 |
| [**BoolQ**](https://arxiv.org/abs/1905.10044) | 0-shot | 63.2 | 72.3 | 78.8 | 82.4 |
| [**PIQA**](https://arxiv.org/abs/1911.11641) | 0-shot | 73.8 | 79.6 | 81.8 | 83.3 |
| [**SocialIQA**](https://arxiv.org/abs/1904.09728) | 0-shot | 48.9 | 51.9 | 53.4 | 54.9 |
| [**TriviaQA**](https://arxiv.org/abs/1705.03551) | 5-shot | 39.8 | 65.8 | 78.2 | 85.5 |
| [**Natural Questions**](https://github.com/google-research-datasets/natural-questions) | 5-shot | 9.48 | 20.0 | 31.4 | 36.1 |
| [**ARC-c**](https://arxiv.org/abs/1911.01547) | 25-shot | 38.4 | 56.2 | 68.9 | 70.6 |
| [**ARC-e**](https://arxiv.org/abs/1911.01547) | 0-shot | 73.0 | 82.4 | 88.3 | 89.0 |
| [**WinoGrande**](https://arxiv.org/abs/1907.10641) | 5-shot | 58.2 | 64.7 | 74.3 | 78.8 |
| [**BIG-Bench Hard**](https://paperswithcode.com/dataset/bbh) | few-shot | 28.4 | 50.9 | 72.6 | 77.7 |
| [**DROP**](https://arxiv.org/abs/1903.00161) | 1-shot | 42.4 | 60.1 | 72.2 | 77.2 |

### microsoft/Phi-4-mini-instruct(3.8B)

https://huggingface.co/microsoft/Phi-4-mini-instruct

Phi-4-mini-instruct is a lightweight open model built upon synthetic data and filtered publicly available websites - with a focus on high-quality, reasoning dense data. The model belongs to the Phi-4 model family and supports 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning and direct preference optimization to support precise instruction adherence and robust safety measures.

| Benchmark | Similar size |  |  |  |  | 2x size |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Phi-4 mini-Ins | Phi-3.5-mini-Ins | Llama-3.2-3B-Ins | Mistral-3B | Qwen2.5-3B-Ins | Qwen2.5-7B-Ins | Mistral-8B-2410 | Llama-3.1-8B-Ins | Llama-3.1-Tulu-3-8B | Gemma2-9B-Ins | GPT-4o-mini-2024-07-18 |
| **Popular aggregated benchmark** |  |  |  |  |  |  |  |  |  |  |  |
| Arena Hard | 32.8 | 34.4 | 17.0 | 26.9 | 32.0 | 55.5 | 37.3 | 25.7 | 42.7 | 43.7 | 53.7 |
| BigBench Hard (0-shot, CoT) | 70.4 | 63.1 | 55.4 | 51.2 | 56.2 | 72.4 | 53.3 | 63.4 | 55.5 | 65.7 | 80.4 |
| MMLU (5-shot) | 67.3 | 65.5 | 61.8 | 60.8 | 65.0 | 72.6 | 63.0 | 68.1 | 65.0 | 71.3 | 77.2 |
| MMLU-Pro (0-shot, CoT) | 52.8 | 47.4 | 39.2 | 35.3 | 44.7 | 56.2 | 36.6 | 44.0 | 40.9 | 50.1 | 62.8 |
| **Reasoning** |  |  |  |  |  |  |  |  |  |  |  |
| ARC Challenge (10-shot) | 83.7 | 84.6 | 76.1 | 80.3 | 82.6 | 90.1 | 82.7 | 83.1 | 79.4 | 89.8 | 93.5 |
| BoolQ (2-shot) | 81.2 | 77.7 | 71.4 | 79.4 | 65.4 | 80.0 | 80.5 | 82.8 | 79.3 | 85.7 | 88.7 |
| GPQA (0-shot, CoT) | 25.2 | 26.6 | 24.3 | 24.4 | 23.4 | 30.6 | 26.3 | 26.3 | 29.9 | 39.1 | 41.1 |
| HellaSwag (5-shot) | 69.1 | 72.2 | 77.2 | 74.6 | 74.6 | 80.0 | 73.5 | 72.8 | 80.9 | 87.1 | 88.7 |
| OpenBookQA (10-shot) | 79.2 | 81.2 | 72.6 | 79.8 | 79.3 | 82.6 | 80.2 | 84.8 | 79.8 | 90.0 | 90.0 |
| PIQA (5-shot) | 77.6 | 78.2 | 68.2 | 73.2 | 72.6 | 76.2 | 81.2 | 83.2 | 78.3 | 83.7 | 88.7 |
| Social IQA (5-shot) | 72.5 | 75.1 | 68.3 | 73.9 | 75.3 | 75.3 | 77.6 | 71.8 | 73.4 | 74.7 | 82.9 |
| TruthfulQA (MC2) (10-shot) | 66.4 | 65.2 | 59.2 | 62.9 | 64.3 | 69.4 | 63.0 | 69.2 | 64.1 | 76.6 | 78.2 |
| Winogrande (5-shot) | 67.0 | 72.2 | 53.2 | 59.8 | 63.3 | 71.1 | 63.1 | 64.7 | 65.4 | 74.0 | 76.9 |
| **Multilingual** |  |  |  |  |  |  |  |  |  |  |  |
| Multilingual MMLU (5-shot) | 49.3 | 51.8 | 48.1 | 46.4 | 55.9 | 64.4 | 53.7 | 56.2 | 54.5 | 63.8 | 72.9 |
| MGSM (0-shot, CoT) | 63.9 | 49.6 | 44.6 | 44.6 | 53.5 | 64.5 | 56.7 | 56.7 | 58.6 | 75.1 | 81.7 |
| **Math** |  |  |  |  |  |  |  |  |  |  |  |
| GSM8K (8-shot, CoT) | 88.6 | 76.9 | 75.6 | 80.1 | 80.6 | 88.7 | 81.9 | 82.4 | 84.3 | 84.9 | 91.3 |
| MATH (0-shot, CoT) | 64.0 | 49.8 | 46.7 | 41.8 | 61.7 | 60.4 | 41.6 | 47.6 | 46.1 | 51.3 | 70.2 |
| **Overall** | **63.5** | **60.5** | **56.2** | **56.9** | **60.1** | **67.9** | **60.2** | **62.3** | **60.9** | **65.0** | **75.5** |

Claude 3.7 Sonnet and Claude Code

https://www.anthropic.com/news/claude-3-7-sonnet

We’ve developed Claude 3.7 Sonnet with a different philosophy from other reasoning models on the market. Just as humans use a single brain for both quick responses and deep reflection, we believe reasoning should be an integrated capability of frontier models rather than a separate model entirely. This unified approach also creates a more seamless experience for users.

![image.png](attachment:23f55435-5c0f-40c9-85d2-90ad16e6caa4:image.png)

### perplexity-ai/r1-1776

https://huggingface.co/perplexity-ai/r1-1776

R1 1776 is a DeepSeek-R1 reasoning model that has been post-trained by Perplexity AI to remove Chinese Communist Party censorship. The model provides unbiased, accurate, and factual information while maintaining high reasoning capabilities.

### mistralai/Mistral-Small-24B-Instruct-2501

https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501

Mistral Small 3 ( 2501 ) sets a new benchmark in the "small" Large Language Models category below 70B, boasting 24B parameters and achieving state-of-the-art capabilities comparable to larger models!

This model is an instruction-fine-tuned version of the base model: [**Mistral-Small-24B-Base-2501**](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501).

Mistral Small can be deployed locally and is exceptionally "knowledge-dense", fitting in a single RTX 4090 or a 32GB RAM MacBook once quantized.

**Human evaluated benchmarks**

| **Category** | **Gemma-2-27B** | **Qwen-2.5-32B** | **Llama-3.3-70B** | **Gpt4o-mini** |
| --- | --- | --- | --- | --- |
| Mistral is better | 0.536 | 0.496 | 0.192 | 0.200 |
| Mistral is slightly better | 0.196 | 0.184 | 0.164 | 0.204 |
| Ties | 0.052 | 0.060 | 0.236 | 0.160 |
| Other is slightly better | 0.060 | 0.088 | 0.112 | 0.124 |
| Other is better | 0.156 | 0.172 | 0.296 | 0.312 |

### deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrated remarkable performance on reasoning. With RL, DeepSeek-R1-Zero naturally emerged with numerous powerful and interesting reasoning behaviors. However, DeepSeek-R1-Zero encounters challenges such as endless repetition, poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks. To support the research community, we have open-sourced DeepSeek-R1-Zero, DeepSeek-R1, and six dense models distilled from DeepSeek-R1 based on Llama and Qwen. DeepSeek-R1-Distill-Qwen-32B outperforms OpenAI-o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.

Distillation: Smaller Models Can Be Powerful Too
We demonstrate that the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models. The open source DeepSeek-R1, as well as its API, will benefit the research community to distill better smaller models in the future.
Using the reasoning data generated by DeepSeek-R1, we fine-tuned several dense models that are widely used in the research community. The evaluation results demonstrate that the distilled smaller dense models perform exceptionally well on benchmarks. We open-source distilled 1.5B, 7B, 8B, 14B, 32B, and 70B checkpoints based on Qwen2.5 and Llama3 series to the community.

| **Model** | **AIME 2024 pass@1** | **AIME 2024 cons@64** | **MATH-500 pass@1** | **GPQA Diamond pass@1** | **LiveCodeBench pass@1** | **CodeForces rating** |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | **1820** |
| QwQ-32B-Preview | 44.0 | 60.0 | 90.6 | 54.5 | 41.9 | 1316 |
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| DeepSeek-R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| DeepSeek-R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| DeepSeek-R1-Distill-Qwen-32B | **72.6** | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| DeepSeek-R1-Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| DeepSeek-R1-Distill-Llama-70B | 70.0 | **86.7** | **94.5** | **65.2** | **57.5** | 1633 |

### MiniMaxAI/MiniMax-Text-01

https://huggingface.co/MiniMaxAI/MiniMax-Text-01

MiniMax-Text-01 is a powerful language model with 456 billion total parameters, of which 45.9 billion are activated per token. To better unlock the long context capabilities of the model, MiniMax-Text-01 adopts a hybrid architecture that combines Lightning Attention, Softmax Attention and Mixture-of-Experts (MoE). Leveraging advanced parallel strategies and innovative compute-communication overlap methods—such as Linear Attention Sequence Parallelism Plus (LASP+), varlen ring attention, Expert Tensor Parallel (ETP), etc., MiniMax-Text-01's training context length is extended to 1 million tokens, and it can handle a context of up to 4 million tokens during the inference. On various academic benchmarks, MiniMax-Text-01 also demonstrates the performance of a top-tier model.

![](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/resolve/main/figures/TextBench.png)

### kyutai/helium-1-preview-2b

https://huggingface.co/kyutai/helium-1-preview-2b

Helium-1 preview is a lightweight language model with 2B parameters, targeting edge and mobile devices. It supports the following languages: English, French, German, Italian, Portuguese, Spanish.

Helium-1 Preview is a base model, which was not fine-tuned to follow instructions or human preferences. For most downstream use cases, the model should be aligned with supervised fine-tuning, RLHF or related methods.

| **Benchmark** | **Helium-1 Preview** | **HF SmolLM2 (1.7B)** | **Gemma-2 (2.6B)** | **Llama-3.2 (3B)** | **Qwen2.5 (1.5B)** |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
| MMLU | 51.2 | 50.4 | 53.1 | 56.6 | 61.0 |
| NQ | 17.3 | 15.1 | 17.7 | 22.0 | 13.1 |
| TQA | 47.9 | 45.4 | 49.9 | 53.6 | 35.9 |
| ARC E | 80.9 | 81.8 | 81.1 | 84.6 | 89.7 |
| ARC C | 62.7 | 64.7 | 66.0 | 69.0 | 77.2 |
| OBQA | 63.8 | 61.4 | 64.6 | 68.4 | 73.8 |
| CSQA | 65.6 | 59.0 | 64.4 | 65.4 | 72.4 |
| PIQA | 77.4 | 77.7 | 79.8 | 78.9 | 76.0 |
| SIQA | 64.4 | 57.5 | 61.9 | 63.8 | 68.7 |
| HS | 69.7 | 73.2 | 74.7 | 76.9 | 67.5 |
| WG | 66.5 | 65.6 | 71.2 | 72.0 | 64.8 |
|  |  |  |  |  |  |
| Average | 60.7 | 59.3 | 62.2 | 64.7 | 63.6 |

### internlm/internlm3-8b-instruct

https://huggingface.co/internlm/internlm3-8b-instruct

InternLM3 has open-sourced an 8-billion parameter instruction model, InternLM3-8B-Instruct, designed for general-purpose usage and advanced reasoning. This model has the following characteristics:

- **Enhanced performance at reduced cost**: State-of-the-art performance on reasoning and knowledge-intensive tasks surpass models like Llama3.1-8B and Qwen2.5-7B. Remarkably, InternLM3 is trained on only 4 trillion high-quality tokens, saving more than 75% of the training cost compared to other LLMs of similar scale.
- **Deep thinking capability**: InternLM3 supports both the deep thinking mode for solving complicated reasoning tasks via the long chain-of-thought and the normal response mode for fluent user interactions.

| **Benchmark** | **InternLM3-8B-Instruct** | **Qwen2.5-7B-Instruct** | **Llama3.1-8B-Instruct** | **GPT-4o-mini(closed source)** |
| --- | --- | --- | --- | --- |
| General | CMMLU(0-shot) | **83.1** | 75.8 | 53.9 |
|  | MMLU(0-shot) | 76.6 | **76.8** | 71.8 |
|  | MMLU-Pro(0-shot) | **57.6** | 56.2 | 48.1 |
| Reasoning | GPQA-Diamond(0-shot) | **37.4** | 33.3 | 24.2 |
|  | DROP(0-shot) | **83.1** | 80.4 | 81.6 |
|  | HellaSwag(10-shot) | **91.2** | 85.3 | 76.7 |
|  | KOR-Bench(0-shot) | **56.4** | 44.6 | 47.7 |
| MATH | MATH-500(0-shot) | **83.0*** | 72.4 | 48.4 |
|  | AIME2024(0-shot) | **20.0*** | 16.7 | 6.7 |
| Coding | LiveCodeBench(2407-2409 Pass@1) | **17.8** | 16.8 | 12.9 |
|  | HumanEval(Pass@1) | 82.3 | **85.4** | 72.0 |
| Instrunction | IFEval(Prompt-Strict) | **79.3** | 71.7 | 75.2 |
| Long Context | RULER(4-128K Average) | 87.9 | 81.4 | **88.5** |
| Chat | AlpacaEval 2.0(LC WinRate) | **51.1** | 30.3 | 25.0 |
|  | WildBench(Raw Score) | **33.1** | 23.3 | 1.5 |
|  | MT-Bench-101(Score 1-10) | **8.59** | 8.49 | 8.37 |

### microsoft/phi-4

https://huggingface.co/microsoft/phi-4

phi-4 is a state-of-the-art open model built upon a blend of synthetic datasets, data from filtered public domain websites, and acquired academic books and Q&A datasets. The goal of this approach was to ensure that small capable models were trained with data focused on high quality and advanced reasoning.

phi-4 underwent a rigorous enhancement and alignment process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures

| **Category** | **Benchmark** | **phi-4 (14B)** | **phi-3 (14B)** | **Qwen 2.5 (14B instruct)** | **GPT-4o-mini** | **Llama-3.3 (70B instruct)** | **Qwen 2.5 (72B instruct)** | **GPT-4o** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popular Aggregated Benchmark | MMLU | 84.8 | 77.9 | 79.9 | 81.8 | 86.3 | 85.3 | **88.1** |
| Science | GPQA | **56.1** | 31.2 | 42.9 | 40.9 | 49.1 | 49.0 | 50.6 |
| Math | MGSMMATH | 80.6**80.4** | 53.544.6 | 79.675.6 | 86.573.0 | 89.166.3* | 87.380.0 | **90.4**74.6 |
| Code Generation | HumanEval | 82.6 | 67.8 | 72.1 | 86.2 | 78.9* | 80.4 | **90.6** |
| Factual Knowledge | SimpleQA | 3.0 | 7.6 | 5.4 | 9.9 | 20.9 | 10.2 | **39.4** |
| Reasoning | DROP | 75.5 | 68.3 | 85.5 | 79.3 | **90.2** | 76.7 | 80.9 |

### deepseek-ai/DeepSeek-V3

https://huggingface.co/deepseek-ai/DeepSeek-V3

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks.

![](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/figures/benchmark.png)

### answerdotai/ModernBERT-large

https://huggingface.co/answerdotai/ModernBERT-large

ModernBERT is a modernized bidirectional encoder-only Transformer model (BERT-style) pre-trained on 2 trillion tokens of English and code data with a native context length of up to 8,192 tokens. ModernBERT leverages recent architectural improvements such as:

- **Rotary Positional Embeddings (RoPE)** for long-context support.
- **Local-Global Alternating Attention** for efficiency on long inputs.
- **Unpadding and Flash Attention** for efficient inference.

ModernBERT’s native long context length makes it ideal for tasks that require processing long documents, such as retrieval, classification, and semantic search within large corpora. The model was trained on a large corpus of text and code, making it suitable for a wide range of downstream tasks, including code retrieval and hybrid (text + code) semantic search.

| **Model** | **IR (DPR)** | **IR (DPR)** | **IR (DPR)** | **IR (ColBERT)** | **IR (ColBERT)** | **NLU** | **Code** | **Code** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | BEIR | MLDR_OOD | MLDR_ID | BEIR | MLDR_OOD | GLUE | CSN | SQA |
| BERT | 38.9 | 23.3 | 31.7 | 49.5 | 28.5 | 85.2 | 41.6 | 60.8 |
| RoBERTa | 41.4 | 22.6 | 36.1 | 49.8 | 28.8 | 88.9 | 47.3 | 68.1 |
| DeBERTaV3 | 25.6 | 7.1 | 19.2 | 46.7 | 23.0 | **91.4** | 21.2 | 19.7 |
| GTE-en-MLM | 42.5 | **36.4** | **48.9** | 50.7 | 71.3 | 87.6 | 40.5 | 66.9 |
| ModernBERT | **44.0** | 34.3 | 48.6 | **52.4** | **80.4** | 90.4 | **59.5** | **83.9** |

### ibm-granite/granite-3.1-8b-instruct

https://huggingface.co/ibm-granite/granite-3.1-8b-instruct

Granite-3.1-8B-Instruct is a 8B parameter long-context instruct model finetuned from Granite-3.1-8B-Base using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets tailored for solving long context problems. This model is developed using a diverse set of techniques with a structured chat format, including supervised finetuning, model alignment using reinforcement learning, and model merging.

### CohereForAI/c4ai-command-r7b-12-2024

https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024

C4AI Command R7B is an open weights research release of a 7B billion parameter model with advanced capabilities optimized for a variety of use cases including reasoning, summarization, question answering, and code. The model is trained to perform sophisticated tasks including Retrieval Augmented Generation (RAG) and tool use. The model also has powerful agentic capabilities with the ability to use and combine multiple tools over multiple steps to accomplish more difficult tasks. It obtains top performance on enterprise relevant code use cases. C4AI Command R7B is a multilingual model trained on 23 languages.

| **Command R7B** | **Gemma 2 IT 9B** | **Ministral 8B** | **Llama 3.1 8B** |
| --- | --- | --- | --- |
| Average | **31.4** | 28.9 | 22 |
| IFEval | 77.9 | 74.4 | 58.96 |
| BBH | 36.1 | **42.1** | 25.82 |
| MATH hard | **26.4** | 0.2 | 6.5 |
| GPQA | 7.7 | **14.8** | 4.5 |
| MUSR | **11.6** | 9.74 | 10.7 |
| MMLU-Pro | 28.5 | **32** | 25.5 |

### meta-llama/Llama-3.3-70B-Instruct

https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.

### utter-project/EuroLLM-9B-Instruct

https://huggingface.co/utter-project/EuroLLM-9B-Instruct

This is the model card for EuroLLM-9B-Instruct. You can also check the pre-trained version: [**EuroLLM-9B**](https://huggingface.co/utter-project/EuroLLM-9B).

- **Developed by:** Unbabel, Instituto Superior Técnico, Instituto de Telecomunicações, University of Edinburgh, Aveni, University of Paris-Saclay, University of Amsterdam, Naver Labs, Sorbonne Université.
- **Funded by:** European Union.
- **Model type:** A 9B parameter multilingual transfomer LLM.
- **Language(s) (NLP):** Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Irish, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Arabic, Catalan, Chinese, Galician, Hindi, Japanese, Korean, Norwegian, Russian, Turkish, and Ukrainian.
- **License:** Apache License 2.0.

### Qwen/QwQ-32B-Preview

https://huggingface.co/Qwen/QwQ-32B-Preview

**QwQ-32B-Preview** is an experimental research model developed by the Qwen Team, focused on advancing AI reasoning capabilities. As a preview release, it demonstrates promising analytical abilities while having several important limitations:

1. **Language Mixing and Code-Switching**: The model may mix languages or switch between them unexpectedly, affecting response clarity.
2. **Recursive Reasoning Loops**: The model may enter circular reasoning patterns, leading to lengthy responses without a conclusive answer.
3. **Safety and Ethical Considerations**: The model requires enhanced safety measures to ensure reliable and secure performance, and users should exercise caution when deploying it.
4. **Performance and Benchmark Limitations**: The model excels in math and coding but has room for improvement in other areas, such as common sense reasoning and nuanced language understanding.

### nvidia/Hymba-1.5B-Base

https://huggingface.co/nvidia/Hymba-1.5B-Base

Hymba-1.5B-Base is a base text-to-text model that can be adopted for a variety of natural language generation tasks.

The model has hybrid architecture with Mamba and Attention heads running in parallel. Meta tokens, a set of learnable tokens prepended to every prompt, help improve the efficacy of the model. The model shares KV cache between 2 layers and between heads in a single layer. 90% of attention layers are sliding window attention.

![](https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/images/performance2.png)

### mistralai/Mistral-Large-Instruct-2411

https://huggingface.co/mistralai/Mistral-Large-Instruct-2411

Mistral-Large-Instruct-2411 is an advanced dense Large Language Model (LLM) of 123B parameters with state-of-the-art reasoning, knowledge and coding capabilities extending [**Mistral-Large-Instruct-2407**](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) with better Long Context, Function Calling and System Prompt.

### Grok from xAI API

https://x.ai/api

xAI provides a powerful developer toolkit for creating scalable applications with its Grok model, which supports diverse tasks like text generation, data extraction, and function calling. To get started, create an account, generate an API key, and explore Grok’s capabilities through available endpoints and integrations.

### tencent/Tencent-Hunyuan-Large

https://huggingface.co/tencent/Tencent-Hunyuan-Large

With the rapid development of artificial intelligence technology, large language models (LLMs) have made significant progress in fields such as natural language processing, computer vision, and scientific tasks. However, as the scale of these models increases, optimizing resource consumption while maintaining high performance has become a key challenge. To address this challenge, we have explored Mixture of Experts (MoE) models. The currently unveiled Hunyuan-Large (Hunyuan-MoE-A52B) model is the largest open-source Transformer-based MoE model in the industry, featuring a total of 389 billion parameters and 52 billion active parameters. This is currently the largest open-source Transformer-based MoE model in the industry, featuring a total of 389 billion parameters and 52 billion active parameters.

| **Model** | **LLama3.1-405B** | **LLama3.1-70B** | **Mixtral-8x22B** | **DeepSeek-V2** | **Hunyuan-Large** |
| --- | --- | --- | --- | --- | --- |
| MMLU | 85.2 | 79.3 | 77.8 | 78.5 | **88.4** |
| MMLU-Pro | **61.6** | 53.8 | 49.5 | - | 60.2 |
| BBH | 85.9 | 81.6 | 78.9 | 78.9 | **86.3** |
| HellaSwag | - | - | **88.7** | 87.8 | 86.8 |
| CommonsenseQA | 85.8 | 84.1 | 82.4 | - | **92.9** |
| WinoGrande | 86.7 | 85.3 | 85.0 | 84.9 | **88.7** |
| PIQA | - | - | 83.6 | 83.7 | **88.3** |
| NaturalQuestions | - | - | 39.6 | 38.7 | **52.8** |
| DROP | 84.8 | 79.6 | 80.4 | 80.1 | **88.9** |
| ARC-C | **96.1** | 92.9 | 91.2 | 92.4 | 95.0 |
| TriviaQA | - | - | 82.1 | 79.9 | **89.2** |
| CMMLU | - | - | 60.0 | 84.0 | **90.2** |
| C-Eval | - | - | 59.6 | 81.7 | **91.9** |
| C3 | - | - | 71.4 | 77.4 | **82.3** |
| GSM8K | 89.0 | 83.7 | 83.7 | 79.2 | **92.8** |
| MATH | 53.8 | 41.4 | 42.5 | 43.6 | **69.8** |
| CMATH | - | - | 72.3 | 78.7 | **91.3** |
| HumanEval | 61.0 | 58.5 | 53.1 | 48.8 | **71.4** |
| MBPP | **73.4** | 68.6 | 64.2 | 66.6 | 72.6 |

### CohereForAI/aya-expanse-32b

https://huggingface.co/CohereForAI/aya-expanse-32b

**Aya Expanse 32B** is an open-weight research release of a model with highly advanced multilingual capabilities. It focuses on pairing a highly performant pre-trained [**Command family**](https://huggingface.co/CohereForAI/c4ai-command-r-plus) of models with the result of a year’s dedicated research from [**Cohere For AI**](https://cohere.for.ai/), including [**data arbitrage**](https://arxiv.org/pdf/2408.14960), [**multilingual preference training**](https://arxiv.org/abs/2407.02552), [**safety tuning**](https://arxiv.org/abs/2406.18682), and [**model merging**](https://arxiv.org/abs/2410.10801). The result is a powerful multilingual large language model serving 23 languages.

![](https://huggingface.co/CohereForAI/aya-expanse-32b/media/main/winrates_marenahard_complete.png)

### HuggingFaceTB/SmolLM2-1.7B-Instruct

https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct

SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.

The 1.7B variant demonstrates significant advances over its predecessor SmolLM1-1.7B, particularly in instruction following, knowledge, reasoning, and mathematics. It was trained on 11 trillion tokens using a diverse dataset combination: FineWeb-Edu, DCLM, The Stack, along with new mathematics and coding datasets that we curated and will release soon. We developed the instruct version through supervised fine-tuning (SFT) using a combination of public datasets and our own curated datasets. We then applied Direct Preference Optimization (DPO) using [**UltraFeedback**](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

| **Metric** | **SmolLM2-1.7B-Instruct** | **Llama-1B-Instruct** | **Qwen2.5-1.5B-Instruct** | **SmolLM1-1.7B-Instruct** |
| --- | --- | --- | --- | --- |
| IFEval (Average prompt/inst) | **56.7** | 53.5 | 47.4 | 23.1 |
| MT-Bench | 6.13 | 5.48 | **6.52** | 4.33 |
| OpenRewrite-Eval (micro_avg RougeL) | 44.9 | 39.2 | **46.9** | NaN |
| HellaSwag | **66.1** | 56.1 | 60.9 | 55.5 |
| ARC (Average) | **51.7** | 41.6 | 46.2 | 43.7 |
| PIQA | **74.4** | 72.3 | 73.2 | 71.6 |
| MMLU-Pro (MCF) | 19.3 | 12.7 | **24.2** | 11.7 |
| BBH (3-shot) | 32.2 | 27.6 | **35.3** | 25.7 |
| GSM8K (5-shot) | **48.2** | 26.8 | 42.8 | 4.62 |

### ibm-granite/granite-3.0-8b-instruct

https://huggingface.co/ibm-granite/granite-3.0-8b-instruct

Granite-3.0-8B-Instruct is a 8B parameter model finetuned from Granite-3.0-8B-Base using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets. This model is developed using a diverse set of techniques with a structured chat format, including supervised finetuning, model alignment using reinforcement learning, and model merging.

### CohereForAI/aya-expanse-8b

https://huggingface.co/CohereForAI/aya-expanse-8b

Aya Expanse is an open-weight research release of a model with highly advanced multilingual capabilities. It focuses on pairing a highly performant pre-trained Command family of models with the result of a year’s dedicated research from Cohere For AI, including data arbitrage, multilingual preference training, safety tuning, and model merging. The result is a powerful multilingual large language model serving 23 languages.

![](https://huggingface.co/CohereForAI/aya-expanse-8b/media/main/winrates_marenahard_complete.png)

### meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8

https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/

- Today, we’re releasing [our first lightweight quantized Llama models](https://llama.com/) that are small and performant enough to run on many popular mobile devices. At Meta, we’re uniquely positioned to provide quantized models because of access to compute resources, training data, full evaluations, and safety.
- As our first quantized models in this Llama category, these instruction-tuned models apply the same quality and safety requirements as the original 1B and 3B models, while achieving 2-4x speedup. We also achieve an average reduction of 56% in model size and a 41% average reduction in memory usage compared to the original BF16 format.
- We used two techniques for quantizing Llama 3.2 1B and 3B models: Quantization-Aware Training with LoRA adaptors, which prioritize accuracy, and SpinQuant, a state-of-the-art post-training quantization method that prioritizes portability.
- Inferences using both quantization techniques are supported in the [Llama Stack](https://github.com/meta-llama/llama-stack) reference implementation via [PyTorch’s ExecuTorch framework](https://github.com/pytorch/executorch).
- We built these quantized models in close collaboration with our industry-leading partners and are making them available on Qualcomm and MediaTek SoCs with Arm CPUs.

### mistralai/Ministral-8B-Instruct-2410

https://huggingface.co/mistralai/Ministral-8B-Instruct-2410

Ministral 8B Key features
Released under the Mistral Research License, reach out to us for a commercial license
Trained with a 128k context window with interleaved sliding-window attention
Trained on a large proportion of multilingual and code data
Supports function calling
Vocabulary size of 131k, using the V3-Tekken tokenizer

Chat/Arena (gpt-4o judge)

| **Model** | **MTBench** | **Arena Hard** | **Wild bench** |
| --- | --- | --- | --- |
| Mistral 7B Instruct v0.3 | 6.7 | 44.3 | 33.1 |
| Llama 3.1 8B Instruct | 7.5 | 62.4 | 37.0 |
| Gemma 2 9B Instruct | 7.6 | 68.7 | ***43.8*** |
| ***Ministral 8B Instruct*** | ***8.3*** | ***70.9*** | 41.3 |
|  |  |  |  |
| Gemma 2 2B Instruct | 7.5 | 51.7 | 32.5 |
| Llama 3.2 3B Instruct | 7.2 | 46.0 | 27.2 |
| ***Ministral 3B Instruct*** | ***8.1*** | ***64.3*** | ***36.3*** |

### nvidia/Llama-3.1-Nemotron-70B-Instruct-HF

https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF

Llama-3.1-Nemotron-70B-Instruct is a large language model customized by NVIDIA to improve the helpfulness of LLM generated responses to user queries.

This model was trained using RLHF (specifically, REINFORCE), [**Llama-3.1-Nemotron-70B-Reward**](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) and [**HelpSteer2-Preference prompts**](https://huggingface.co/datasets/nvidia/HelpSteer2) on a Llama-3.1-70B-Instruct model as the initial policy.
Try hosted inference for free at [**build.nvidia.com**](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct) - it comes with an OpenAI-compatible API interface.

| **Model** | **Arena Hard** | **AlpacaEval** | **MT-Bench** | **Mean Response Length** |
| --- | --- | --- | --- | --- |
| Details | (95% CI) | 2 LC (SE) | (GPT-4-Turbo) | (# of Characters for MT-Bench) |
| ***Llama-3.1-Nemotron-70B-Instruct*** | **85.0** (-1.5, 1.5) | **57.6** (1.65) | **8.98** | 2199.8 |
| Llama-3.1-70B-Instruct | 55.7 (-2.9, 2.7) | 38.1 (0.90) | 8.22 | 1728.6 |
| Llama-3.1-405B-Instruct | 69.3 (-2.4, 2.2) | 39.3 (1.43) | 8.49 | 1664.7 |
| Claude-3-5-Sonnet-20240620 | 79.2 (-1.9, 1.7) | 52.4 (1.47) | 8.81 | 1619.9 |
| GPT-4o-2024-05-13 | 79.3 (-2.1, 2.0) | 57.5 (1.47) | 8.74 | 1752.2 |

### Zyphra/Zamba2-7B

https://huggingface.co/Zyphra/Zamba2-7B

Zamba2-7B is a hybrid model combining state-space (Mamba2) and transformer blocks, based on the Zamba architecture. Key improvements over Zamba1 include replacing Mamba1 with Mamba2 blocks, adding LoRA projectors for depth-specialization with minimal parameter increase, using two alternating shared attention blocks, and applying rotary position embeddings to enhance performance. Pre-trained on 2T tokens of text and code, Zamba2-7B was later refined on 100B high-quality tokens. Currently, it's a temporary HuggingFace implementation and may not be fully compatible with all tools.

![](https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/nHM8bX0y8SWa4zwMSbBi7.png)

![](https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/qXG8aip6h77LHKjhWfjD5.png)

### allenai/Molmo-7B-D-0924

https://huggingface.co/allenai/Molmo-7B-D-0924

Molmo is a family of open vision-language models developed by the Allen Institute for AI. Molmo models are trained on PixMo, a dataset of 1 million, highly-curated image-text pairs. It has state-of-the-art performance among multimodal models with a similar size while being fully open-source. You can find all models in the Molmo family [**here**](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19). **Learn more** about the Molmo family [**in our announcement blog post**](https://molmo.allenai.org/blog) or the [**paper**](https://huggingface.co/papers/2409.17146).

Molmo 7B-D is based on [**Qwen2-7B**](https://huggingface.co/Qwen/Qwen2-7B) and uses [**OpenAI CLIP**](https://huggingface.co/openai/clip-vit-large-patch14-336) as vision backbone. It performs comfortably between GPT-4V and GPT-4o on both academic benchmarks and human evaluation. It powers the **Molmo demo at** [**molmo.allenai.org**](https://molmo.allenai.org/).

**Evaluations**

| **Model** | **Average Score on 11 Academic Benchmarks** | **Human Preference Elo Rating** |
| --- | --- | --- |
| Molmo 72B | 81.2 | 1077 |
| **Molmo 7B-D (this model)** | **77.3** | **1056** |
| Molmo 7B-O | 74.6 | 1051 |
| MolmoE 1B | 68.6 | 1032 |
| GPT-4o | 78.5 | 1079 |
| GPT-4V | 71.1 | 1041 |
| Gemini 1.5 Pro | 78.3 | 1074 |
| Gemini 1.5 Flash | 75.1 | 1054 |
| Claude 3.5 Sonnet | 76.7 | 1069 |
| Claude 3 Opus | 66.4 | 971 |
| Claude 3 Haiku | 65.3 | 999 |
| Qwen VL2 72B | 79.4 | 1037 |
| Qwen VL2 7B | 73.7 | 1025 |
| Intern VL2 LLAMA 76B | 77.1 | 1018 |
| Intern VL2 8B | 69.4 | 953 |
| Pixtral 12B | 69.5 | 1016 |
| Phi3.5-Vision 4B | 59.7 | 982 |
| PaliGemma 3B | 50.0 | 937 |
| LLAVA OneVision 72B | 76.6 | 1051 |
| LLAVA OneVision 7B | 72.0 | 1024 |
| Cambrian-1 34B | 66.8 | 953 |
| Cambrian-1 8B | 63.4 | 952 |
| xGen - MM - Interleave 4B | 59.5 | 979 |
| LLAVA-1.5 13B | 43.9 | 960 |
| LLAVA-1.5 7B | 40.7 | 951 |

Benchmarks: AI2D test, ChartQA test, VQA v2.0 test, DocQA test, InfographicVQA test, TextVQA val, RealWorldQA, MMMU val, MathVista testmini, CountBenchQA, Flickr Count (we collected this new dataset that is significantly harder than CountBenchQA).

### rhymes-ai/Aria

https://huggingface.co/rhymes-ai/Aria

- **SoTA Multimodal Native Performance**: Aria achieves strong performance on a wide range of multimodal, language, and coding tasks. It is superior in video and document understanding.
- **Lightweight and Fast**: Aria is a mixture-of-expert model with 3.9B activated parameters per token. It efficently encodes visual input of variable sizes and aspect ratios.
- **Long Multimodal Context Window**: Aria supports multimodal input of up to 64K tokens. It can caption a 256-frame video in 10 seconds.

| **Category** | **Benchmark** | **Aria** | **Pixtral 12B** | **Llama3.2 11B** | **GPT-4o mini** | **Gemini-1.5 Flash** |
| --- | --- | --- | --- | --- | --- | --- |
| **Knowledge (Multimodal)** | MMMU | 54.9 | 52.5 | 50.7 | 59.4 | 56.1 |
| **Math (Multimodal)** | MathVista | 66.1 | 58.0 | 51.5 | - | 58.4 |
| **Document** | DocQA | 92.6 | 90.7 | 84.4 | - | 89.9 |
| **Chart** | ChartQA | 86.4 | 81.8 | 83.4 | - | 85.4 |
| **Scene Text** | TextVQA | 81.1 | - | - | - | 78.7 |
| **General Visual QA** | MMBench-1.1 | 80.3 | - | - | 76.0 | - |
| **Video Understanding** | LongVideoBench | 65.3 | 47.4 | 45.7 | 58.8 | 62.4 |
| **Knowledge (Language)** | MMLU (5-shot) | 73.3 | 69.2 | 69.4 | - | 78.9 |
| **Math (Language)** | MATH | 50.8 | 48.1 | 51.9 | 70.2 | - |
| **Reasoning (Language)** | ARC Challenge | 91.0 | - | 83.4 | 96.4 | - |
| **Coding** | HumanEval | 73.2 | 72.0 | 72.6 | 87.2 | 74.3 |

### arcee-ai/SuperNova-Medius

https://huggingface.co/arcee-ai/SuperNova-Medius

Arcee-SuperNova-Medius is a 14B parameter language model developed by [Arcee.ai](http://arcee.ai/), built on the Qwen2.5-14B-Instruct architecture. This unique model is the result of a cross-architecture distillation pipeline, combining knowledge from both the Qwen2.5-72B-Instruct model and the Llama-3.1-405B-Instruct model. By leveraging the strengths of these two distinct architectures, SuperNova-Medius achieves high-quality instruction-following and complex reasoning capabilities in a mid-sized, resource-efficient form.

SuperNova-Medius is designed to excel in a variety of business use cases, including customer support, content creation, and technical assistance, while maintaining compatibility with smaller hardware configurations. It’s an ideal solution for organizations looking for advanced capabilities without the high resource requirements of larger models like our SuperNova-70B.

| **Model** | **Average** | **IFEval** | **BBH** | **GPQA** | **MMLU Pro** | **MuSR** | **Math Level 5** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mistral-Small 2409 | 0.423 | 0.628 | 0.581 | 0.333 | 0.410 | 0.406 | 0.181 |
| Supernova-Lite | 0.427 | 0.786 | 0.511 | 0.306 | 0.388 | 0.415 | 0.155 |
| Qwen2.5-14B-Instruct | 0.450 | 0.827 | 0.623 | 0.358 | 0.490 | 0.403 | 0.000 |
| Supernova-Medius | **0.480** | **0.832** | **0.631** | **0.359** | **0.502** | **0.402** | **0.152** |

### Zyphra/Zamba2-2.7B-instruct

https://huggingface.co/Zyphra/Zamba2-2.7B-instruct

Zamba2-2.7B-Instruct is obtained from [**Zamba2-2.7B**](https://huggingface.co/Zyphra/Zamba2-2.7B) by fine-tuning on instruction-following and chat datasets. Specifically:

1. SFT of the base [**Zamba2-2.7B**](https://huggingface.co/Zyphra/Zamba2-2.7B) model on [**ultrachat_200k**](https://huggingface.co/Zyphra/Zamba2-2.7B-instruct/blob/main/HuggingFaceH4/ultrachat_200k) and [**Infinity-Instruct**](https://huggingface.co/datasets/BAAI/Infinity-Instruct)
2. DPO of the SFT checkpoint on [**ultrafeedback_binarized**](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), [**orca_dpo_pairs**](https://huggingface.co/datasets/Intel/orca_dpo_pairs), and [**OpenHermesPreferences**](https://huggingface.co/datasets/argilla/OpenHermesPreferences)

Zamba2-2.7B-Instruct is a hybrid model composed of state-space ([**Mamba2**](https://github.com/state-spaces/mamba)) and transformer blocks.

![](https://cdn-uploads.huggingface.co/production/uploads/65bc13717c6ad1994b6619e9/QnudHrMeMx_NuRc2evwRG.png)

### KingNish/Reasoning-Llama-1b-v0.1

https://huggingface.co/KingNish/Reasoning-Llama-1b-v0.1

It's First iteration of this model. For testing purpose its just trained on 10k rows. It performed very well than expected. It do first reasoning and than generate response on based on it but it do like o1. It do reasoning separately (Just like o1), no tags (like reflection). 

Hongsu’s test:

```jsx
EASONING: To determine which is greater, 9.9 or 9.11, we need to consider the decimal places and the digits in each number.

**Understanding the Problem**

We are given two decimal numbers: 9.9 and 9.11. Our goal is to compare these two numbers and determine which one is greater.

**Breaking Down the Problem**

To compare these numbers, we can start by examining the decimal places. Both numbers have two decimal places, which means we need to consider the digits in each place.

**Analyzing the Digits**

Let's analyze the digits in each number:

* 9.9 has two digits: 9 and 9.
* 9.11 has two digits: 9 and 1.

**Comparing the Digits**

Since both numbers have two digits, we can compare the digits in each place. The first digit is the same for both numbers, so we need to focus on the second digit.

* 9.9 has a 9 in the second place.
* 9.11 has a 1 in the second place.

**Comparing the Second Digit**

Now, let's compare the second digits: 9 and 1. Since 1 is less than 9, we can conclude that 9.9 is greater than 9.11.

**Conclusion**

Based on the analysis, we can conclude that 9.9 is greater than 9.11.

**Self-Reflection**

* We started by breaking down the problem and analyzing the digits in each number.
* We compared the digits in each place, focusing on the second digit.
* We concluded that 9.9 is greater than 9.11 based on the comparison of the second digits.

**Advanced Reasoning**

To further solidify our understanding, let's consider the concept of decimal places and the significance of the second digit.

* A decimal place is a place in a decimal number that represents a fraction of 1.
* The second digit in a decimal number represents the hundredths place.
* When comparing decimal numbers, we need to consider the decimal places and the significance of the second digit.

**Advanced Reasoning Example**

Let's consider a more advanced example to illustrate the concept of decimal places and the significance of the second digit.

* Suppose we have the decimal number 0.54321.
* We can analyze the digits in this number: 5, 4, 3, and 2.
* The first digit is the same for both numbers, so we need to focus on the second digit.
* The second digit is the 5 in the second place.
* Since the second digit is 5, we can conclude that 0.54321 is greater than 0.5231.

**Advanced Reasoning Conclusion**

Based on the advanced reasoning, we can conclude that 0.54321 is greater than 0.5231.

**Final Answer**

Based on our analysis and advanced reasoning, we can conclude that 9.9 is greater than 9.11.
ANSWER: 9.9 is greater than 9.11 because the second digit in 9.9 is 9, while the second digit in 9.11 is 1.

```

### nvidia/NVLM-D-72B

https://huggingface.co/nvidia/NVLM-D-72B

Today (September 17th, 2024), we introduce NVLM 1.0, a family of frontier-class multimodal large language models (LLMs) that achieve state-of-the-art results on vision-language tasks, rivaling the leading proprietary models (e.g., GPT-4o) and open-access models (e.g., Llama 3-V 405B and InternVL 2). Remarkably, NVLM 1.0 shows improved text-only performance over its LLM backbone after multimodal training.

Results (as of September 17th, 2024) in the multimodal benchmarks are as follows:

| **Benchmark** | **MMMU (val / test)** | **MathVista** | **OCRBench** | **AI2D** | **ChartQA** | **DocVQA** | **TextVQA** | **RealWorldQA** | **VQAv2** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVLM-D 1.0 72B (Huggingface) | 58.7 / 54.9 | 65.2 | 852 | 94.2 | 86.0 | 92.6 | 82.6 | 69.5 | 85.4 |
| NVLM-D 1.0 72B (Megatron) | 59.7 / 54.6 | 65.2 | 853 | 94.2 | 86.0 | 92.6 | 82.1 | 69.7 | 85.4 |
| Llama 3.2 90B | 60.3 / - | 57.3 | - | 92.3 | 85.5 | 90.1 | - | - | 78.1 |
| Llama 3-V 70B | 60.6 / - | - | - | 93.0 | 83.2 | 92.2 | 83.4 | - | 79.1 |
| Llama 3-V 405B | 64.5 / - | - | - | 94.1 | 85.8 | 92.6 | 84.8 | - | 80.2 |
| InternVL2-Llama3-76B | 55.2 / - | 65.5 | 839 | 94.8 | 88.4 | 94.1 | 84.4 | 72.2 | - |
| GPT-4V | 56.8 / 55.7 | 49.9 | 645 | 78.2 | 78.5 | 88.4 | 78.0 | 61.4 | 77.2 |
| GPT-4o | 69.1 / - | 63.8 | 736 | 94.2 | 85.7 | 92.8 | - | - | - |
| Claude 3.5 Sonnet | 68.3 / - | 67.7 | 788 | 94.7 | 90.8 | 95.2 | - | - | - |
| Gemini 1.5 Pro (Aug 2024) | 62.2 / - | 63.9 | 754 | 94.4 | 87.2 | 93.1 | 78.7 | 70.4 | 80.2 |

### allenai/Molmo-72B-0924

https://huggingface.co/allenai/Molmo-72B-0924

olmo is a family of open vision-language models developed by the Allen Institute for AI. Molmo models are trained on PixMo, a dataset of 1 million, highly-curated image-text pairs. It has state-of-the-art performance among multimodal models with a similar size while being fully open-source. You can find all models in the Molmo family [**here**](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19). **Learn more** about the Molmo family [**in our announcement blog post**](https://molmo.allenai.org/blog).

Molmo 72B is based on [**Qwen2-72B**](https://huggingface.co/Qwen/Qwen2-72B) and uses [**OpenAI CLIP**](https://huggingface.co/openai/clip-vit-large-patch14-336) as vision backbone. Molmo-72B achieves the highest academic benchmark score and ranks second on human evaluation, just slightly behind GPT-4o.

**Evaluations**

| **Model** | **Average Score on 11 Academic Benchmarks** | **Human Preference Elo Rating** |
| --- | --- | --- |
| **Molmo 72B (this model)** | **81.2** | **1077** |
| Molmo 7B-D | 77.3 | 1056 |
| Molmo 7B-O | 74.6 | 1051 |
| MolmoE 1B | 68.6 | 1032 |
| GPT-4o | 78.5 | 1079 |
| GPT-4V | 71.1 | 1041 |
| Gemini 1.5 Pro | 78.3 | 1074 |
| Gemini 1.5 Flash | 75.1 | 1054 |
| Claude 3.5 Sonnet | 76.7 | 1069 |
| Claude 3 Opus | 66.4 | 971 |
| Claude 3 Haiku | 65.3 | 999 |
| Qwen VL2 72B | 79.4 | 1037 |
| Qwen VL2 7B | 73.7 | 1025 |
| Intern VL2 LLAMA 76B | 77.1 | 1018 |
| Intern VL2 8B | 69.4 | 953 |
| Pixtral 12B | 69.5 | 1016 |
| Phi3.5-Vision 4B | 59.7 | 982 |

### meta-llama/Llama-3.2-11B-Vision-Instruct

https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The models outperform many of the available open source and closed multimodal models on common industry benchmarks.

### mistralai/Mistral-Small-Instruct-2409

https://huggingface.co/mistralai/Mistral-Small-Instruct-2409

Mistral-Small-Instruct-2409 is an instruct fine-tuned version with the following characteristics: 22B parameters, Vocabulary to 32768, Supports function calling, 32k sequence length

### microsoft/GRIN-MoE

https://huggingface.co/microsoft/GRIN-MoE

With only 6.6B activate parameters, GRIN MoE achieves exceptionally good performance across a diverse set of tasks, particularly in coding and mathematics tasks.
GRIN uses SparseMixer-v2 to estimate the gradient related to expert routing, while the conventional MoE training treats expert gating as a proxy for the gradient estimation.
GRIN scales MoE training with neither expert parallelism nor token dropping, while the conventional MoE training employs expert parallelism and deploys token dropping.

Benchmarks:

|  | **GRIN MoE (16x3.8B)** | **Mixtral (8x7B)** | **Mixtral (8x22B)** | **Llama3 (8B)** | **Llama3 (70B)** | **GPT3.5** | **GPT4o** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU | 79.4 | 70.5 | 76.2 | 66.5 | 80.2 | 71.4 | 86.9 |
| HellaSwag | 83.7 | 70.4 | 79.0 | 71.1 | 82.6 | 78.8 | 91.7 |
| ANLI | 60.6 | 55.2 | 65.2 | 57.3 | 68.3 | 58.1 | 75.7 |
| GSM-8K | 90.4 | 64.7 | 83.8 | 77.4 | 93.5 | 78.1 | 93.8 |
| Math | 58.9 | 11.1 | 41.8 | 28.2 | 51.2 | 45.3 | 67.8 |
| MedQA | 70.4 | 62.2 | 67.9 | 60.5 | 78.5 | 63.4 | 88.9 |
| AGIEval | 48.2 | 45.2 | 54.0 | 42.0 | 56.9 | 48.4 | 37.6 |
| TriviaQA | 73.9 | 78.5 | 82.2 | 67.7 | 84.5 | 85.8 | 66.0 |
| Arc-C | 92.0 | 87.3 | 91.3 | 82.8 | 93.0 | 87.4 | 97.0 |
| Arc-E | 98.0 | 95.6 | 96.9 | 93.4 | 98.2 | 96.3 | 99.0 |
| PIQA | 89.0 | 86.0 | 85.0 | 75.7 | 85.3 | 86.6 | 92.9 |
| SociQA | 79.5 | 75.9 | 78.2 | 73.9 | 81.1 | 68.3 | 81.4 |
| BigBench-Hard | 81.4 | 69.7 | 81.8 | 51.5 | 80.2 | 68.3 | 81.2 |
| WinoGrande | 81.4 | 62.0 | 75.3 | 65.0 | 83.3 | 68.8 | 89.3 |
| OpenBookQA | 89.8 | 85.8 | 88.6 | 82.6 | 91.8 | 86.0 | 95.2 |
| BoolQ | 83.4 | 77.6 | 82.7 | 80.9 | 89.1 | 79.1 | 90.6 |
| CommonSenseQA | 81.8 | 78.1 | 82.0 | 79.0 | 84.4 | 79.6 | 88.5 |
| TruthfulQA | 74.5 | 60.1 | 67.4 | 63.2 | 81.9 | 85.8 | 85.6 |
| HumanEval | 74.4 | 37.8 | 39.6 | 60.4 | 78.7 | 62.2 | 92.1 |
| MBPP | 80.3 | 60.2 | 70.7 | 67.7 | 81.3 | 77.8 | 90.4 |
| Average | 78.6 | 66.7 | 74.5 | 67.3 | 81.2 | 73.8 | 84.8 |

### Qwen/Qwen2.5-72B-Instruct

https://huggingface.co/Qwen/Qwen2.5-72B-Instruct

Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters. Qwen2.5 brings the following improvements upon Qwen2:

- Significantly **more knowledge** and has greatly improved capabilities in **coding** and **mathematics**, thanks to our specialized expert models in these domains.
- Significant improvements in **instruction following**, **generating long texts** (over 8K tokens), **understanding structured data** (e.g, tables), and **generating structured outputs** especially JSON. **More resilient to the diversity of system prompts**, enhancing role-play implementation and condition-setting for chatbots.
- **Long-context Support** up to 128K tokens and can generate up to 8K tokens.
- **Multilingual support** for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

Benchmarks:

https://qwenlm.github.io/blog/qwen2.5/

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5/Qwen2.5-72B-Instruct-Score.jpg)

### arcee-ai/Llama-3.1-SuperNova-Lite

https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite

Llama-3.1-SuperNova-Lite is an 8B parameter model developed by [Arcee.ai](http://arcee.ai/), based on the Llama-3.1-8B-Instruct architecture. It is a distilled version of the larger Llama-3.1-405B-Instruct model, leveraging offline logits extracted from the 405B parameter variant. This 8B variation of Llama-3.1-SuperNova maintains high performance while offering exceptional instruction-following capabilities and domain-specific adaptability.

Benchmark

| **Benchmark** | **SuperNova-Lite** | **Llama-3.1-8b-Instruct** |
| --- | --- | --- |
| IF_Eval | 81.1 | 77.4 |
| MMLU Pro | 38.7 | 37.7 |
| TruthfulQA | 64.4 | 55.0 |
| BBH | 51.1 | 50.6 |
| GPQA | 31.2 | 29.02 |

### OpenAI o1

https://openai.com/index/introducing-openai-o1-preview/

The next model update performs similarly to PhD students on challenging benchmark tasks in physics, chemistry, and biology. We also found that it excels in math and coding. In a qualifying exam for the International Mathematics Olympiad (IMO), GPT-4o correctly solved only 13% of problems, while the reasoning model scored 83%. Their coding abilities were evaluated in contests and reached the 89th percentile in Codeforces competitions. You can read more about this in our technical research post.

As an early model, it doesn't yet have many of the features that make ChatGPT useful, like browsing the web for information and uploading files and images. For many common cases GPT-4o will be more capable in the near term.

But for complex reasoning tasks this is a significant advancement and represents a new level of AI capability. Given this, we are resetting the counter back to 1 and naming this series OpenAI o1.

### openbmb/MiniCPM3-4B

https://huggingface.co/openbmb/MiniCPM3-4B

MiniCPM3-4B is the 3rd generation of MiniCPM series. The overall performance of MiniCPM3-4B surpasses Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125, being comparable with many recent 7B~9B models. MiniCPM3-4B has a 32k context window. Equipped with LLMxMapReduce, MiniCPM3-4B can handle infinite context theoretically, without requiring huge amount of memory.

Evaluation Results

| Benchmark | Qwen2-7B-Instruct | GLM-4-9B-Chat | Gemma2-9B-it | Llama3.1-8B-Instruct | GPT-3.5-Turbo-0125 | Phi-3.5-mini-Instruct(3.8B) | MiniCPM3-4B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **English** |  |  |  |  |  |  |  |
| MMLU | 70.5 | 72.4 | 72.6 | 69.4 | 69.2 | 68.4 | 67.2 |
| BBH | 64.9 | 76.3 | 65.2 | 67.8 | 70.3 | 68.6 | 70.2 |
| MT-Bench | 8.41 | 8.35 | 7.88 | 8.28 | 8.17 | 8.60 | 8.41 |
| IFEVAL (Prompt Strict-Acc.) | 51.0 | 64.5 | 71.9 | 71.5 | 58.8 | 49.4 | 68.4 |
| **Chinese** |  |  |  |  |  |  |  |
| CMMLU | 80.9 | 71.5 | 59.5 | 55.8 | 54.5 | 46.9 | 73.3 |
| CEVAL | 77.2 | 75.6 | 56.7 | 55.2 | 52.8 | 46.1 | 73.6 |
| AlignBench v1.1 | 7.10 | 6.61 | 7.10 | 5.68 | 5.82 | 5.73 | 6.74 |
| FollowBench-zh (SSR) | 63.0 | 56.4 | 57.0 | 50.6 | 64.6 | 58.1 | 66.8 |

### upstage/solar-pro-preview-instruct

https://huggingface.co/upstage/solar-pro-preview-instruct

We introduce Solar Pro Preview, an advanced large language model (LLM) with 22 billion parameters designed to fit into a single GPU. Solar Pro Preview shows superior performance compared to LLMs with less than 30 billion parameters and delivers performance comparable to models over three times its size, such as Llama 3.1 with 70 billion parameters.

Solar Pro Preview is developed using an enhanced version of our previous depth up-scaling method, which scales a Phi-3-medium model with 14 billion parameters to 22 billion parameters, intended to run on a GPU with 80GB of VRAM. Our carefully curated training strategy and dataset have significantly enhanced performance from Phi-3-medium, particularly on the MMLU-Pro and IFEval benchmarks, both respected for evaluating a model’s knowledge and instruction-following abilities.

![](https://cdn-uploads.huggingface.co/production/uploads/5fd90c758fe27b1a6b077abb/jwMkqV88Hj8sJu7NjTedm.png)

### deepseek-ai/DeepSeek-V2.5

https://huggingface.co/deepseek-ai/DeepSeek-V2.5

DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. The new model integrates the general and coding abilities of the two previous versions. For model details, please visit DeepSeek-V2 page for more information.

Benchmarks:

| **Metric** | **DeepSeek-V2-0628** | **DeepSeek-Coder-V2-0724** | **DeepSeek-V2.5** |
| --- | --- | --- | --- |
| AlpacaEval 2.0 | 46.6 | 44.5 | 50.5 |
| ArenaHard | 68.3 | 66.3 | 76.2 |
| AlignBench | 7.88 | 7.91 | 8.04 |
| MT-Bench | 8.85 | 8.91 | 9.02 |
| HumanEval python | 84.5 | 87.2 | 89 |
| HumanEval Multi | 73.8 | 74.8 | 73.8 |
| LiveCodeBench(01-09) | 36.6 | 39.7 | 41.8 |
| Aider | 69.9 | 72.9 | 72.2 |
| SWE-verified | N/A | 19 | 16.8 |
| DS-FIM-Eval | N/A | 73.2 | 78.3 |
| DS-Arena-Code | N/A | 49.5 | 63.1 |

### allenai/OLMoE-1B-7B-0924

https://huggingface.co/allenai/OLMoE-1B-7B-0924

OLMoE-1B-7B is a Mixture-of-Experts LLM with 1B active and 7B total parameters released in September 2024 (0924). It yields state-of-the-art performance among models with a similar cost (1B) and is competitive with much larger models like Llama2-13B. OLMoE is 100% open-source.

Evaluation Snapshot:

| **Model** | **Active Params** | **Open Data** | **MMLU** | **HellaSwag** | **ARC-Chall.** | **ARC-Easy** | **PIQA** | **WinoGrande** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LMs with ~1B active parameters** |  |  |  |  |  |  |  |  |
| **OLMoE-1B-7B** | **1.3B** | **✅** | **54.1** | **80.0** | **62.1** | **84.2** | **79.8** | **70.2** |
| DCLM-1B | 1.4B | ✅ | 48.5 | 75.1 | 57.6 | 79.5 | 76.6 | 68.1 |
| TinyLlama-1B | 1.1B | ✅ | 33.6 | 60.8 | 38.1 | 69.5 | 71.7 | 60.1 |
| OLMo-1B (0724) | 1.3B | ✅ | 32.1 | 67.5 | 36.4 | 53.5 | 74.0 | 62.9 |
| Pythia-1B | 1.1B | ✅ | 31.1 | 48.0 | 31.4 | 63.4 | 68.9 | 52.7 |

### mattshumer/Reflection-70B

https://huggingface.co/mattshumer/Reflection-70B

Reflection 70B is (currently) the world's top open-source LLM, trained with a new technique called Reflection-Tuning that teaches a LLM to detect mistakes in its reasoning and correct course.

Benchmarks:

![](https://cdn-uploads.huggingface.co/production/uploads/60518f3731c5be7f3dd5ebc3/zNs-ZFs0SbnomH7mikiOU.png)

### Zyphra/Zamba2-2.7B

https://huggingface.co/Zyphra/Zamba2-2.7B

Zamba2-2.7B is a hybrid model composed of state-space and transformer blocks. It broadly follows the [**Zamba architecture**](https://arxiv.org/abs/2405.16712) which consists of a Mamba backbone alternating with shared transformer blocks (see diagram in [**Model Details**](https://huggingface.co/Zyphra/Zamba2-2.7B#model-details)). 

![](https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/U7VD9PYLj3XcEjgV08sP5.png)

### gpt-omni/mini-omni

https://huggingface.co/gpt-omni/mini-omni

Mini-Omni is an open-source multimodel large language model that can hear, talk while thinking. Featuring real-time end-to-end speech input and streaming audio output conversational capabilities.

### c4ai-command-r-08-2024 and CohereForAI/c4ai-command-r-plus-08-2024

https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024

C4AI Command R+ 08-2024 is an open weights research release of a 104B billion parameter model with highly advanced capabilities, this includes Retrieval Augmented Generation (RAG) and tool use to automate sophisticated tasks. The tool use in this model generation enables multi-step tool use which allows the model to combine multiple tools over multiple steps to accomplish difficult tasks. C4AI Command R+ 08-2024 is a multilingual model trained on 23 languages and evaluated in 10 languages. Command R+ 08-2024 is optimized for a variety of use cases including reasoning, summarization, and question answering.

https://huggingface.co/CohereForAI/c4ai-command-r-08-2024

C4AI Command R 08-2024 is a research release of a 35 billion parameter highly performant generative model. Command R 08-2024 is a large language model with open weights optimized for a variety of use cases including reasoning, summarization, and question answering. Command R 08-2024 has the capability for multilingual generation, trained on 23 languages and evaluated in 10 languages and highly performant RAG capabilities.

**Cohere playgroud**

https://dashboard.cohere.com/playground/chat

### microsoft/Phi-3.5-MoE-instruct

https://huggingface.co/microsoft/Phi-3.5-MoE-instruct

Phi-3.5-MoE is a lightweight, state-of-the-art open model built upon datasets used for Phi-3 - synthetic data and filtered publicly available documents - with a focus on very high-quality, reasoning dense data. The model supports multilingual and comes with 128K context length (in tokens). The model underwent a rigorous enhancement process, incorporating supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures.

Phi-3 MoE has 16x3.8B parameters with 6.6B active parameters when using 2 experts. The model is a mixture-of-expert decoder-only Transformer model using the tokenizer with vocabulary size of 32,064. The model is intended for broad commercial and research use in English. The model provides uses for general purpose AI systems and applications which require

### PleIAs/OCRonos-Vintage

https://huggingface.co/PleIAs/OCRonos-Vintage

OCRonos-Vintage was pre-trained from scratch on a dataset of cultural heritage archives from the Library of Congress, Internet Archive and Hathi Trust totalling 18 billion tokens.

OCRonos-Vintage is only 124 million parameters. It can run easily on CPU or provide correction at scale on GPUs (>10k tokens/seconds) while providing a quality of correction comparable to GPT-4 or the llama version of OCRonos for English-speaking cultural archives.

### HuggingFaceM4/Idefics3-8B-Llama3

https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3

Idefics3 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text outputs. The model can answer questions about images, describe visual content, create stories grounded on multiple images, or simply behave as a pure language model without visual inputs. It improves upon Idefics1 and Idefics2, significantly enhancing capabilities around OCR, document understanding and visual reasoning.

### google/gemma-2-9b

https://huggingface.co/google/gemma-2-9b

Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights for both pre-trained variants and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

Hongsu: On chatbot-arena-leaderboard, gemma-2-9b gets 1218, while GPT4-0314 is 1186(August, 4)

### Meta-Llama-3.1

https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.

Triplex: a SOTA LLM for knowledge graph construction.

https://huggingface.co/SciPhi/Triplex

Triplex is a finetuned version of Phi3-3.8B for creating knowledge graphs from unstructured data developed by SciPhi.AI. It works by extracting triplets - simple statements consisting of a subject, predicate, and object - from text or other data sources.

### mistralai/Mistral-Nemo-Instruct-2407

https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407

Released under the Apache 2 License, Pre-trained and instructed versions, Trained with a 128k context window, Trained on a large proportion of multilingual and code data, Drop-in replacement of Mistral 7B

### GPT-4o mini: advancing cost-efficient intelligence

https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/

OpenAI is committed to making intelligence as broadly accessible as possible. Today, we're announcing GPT-4o mini, our most cost-efficient small model. We expect GPT-4o mini will significantly expand the range of applications built with AI by making intelligence much more affordable. GPT-4o mini scores 82% on MMLU and currently outperforms GPT-41 on chat preferences in LMSYS leaderboard(opens in a new window). It is priced at 15 cents per million input tokens and 60 cents per million output tokens, an order of magnitude more affordable than previous frontier models and more than 60% cheaper than GPT-3.5 Turbo.

### GAIR/Anole-7b-v0.1

https://huggingface.co/GAIR/Anole-7b-v0.1

Anole is the first open-source, autoregressive, and natively trained large multimodal model capable of interleaved image-text generation (without using stable diffusion). 

### facebook/chameleon-7b

https://huggingface.co/facebook/chameleon-7b

Repository for Meta Chameleon, a mixed-modal early-fusion foundation model from FAIR. The Chameleon collection on HuggingFace contains 7 billion parameter and 30 billion parameter model checkpoints.

### Claude 3.5 Sonnet

https://www.anthropic.com/news/claude-3-5-sonnet

### microsoft/Phi-3 medium, small, mini, vision

https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3

### THUDM/glm-4-9b-chat

https://huggingface.co/THUDM/glm-4-9b-chat

我们还推出了支持 1M 上下文长度（约 200 万中文字符）的 GLM-4-9B-Chat-1M 模型和基于 GLM-4-9B 的多模态模型 GLM-4V-9B。

### Qwen2-72B

https://huggingface.co/Qwen/Qwen2-72B

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This repo contains the 72B Qwen2 base language model.

### Mistral-7B-v0.3 has the following changes compared to Mistral-7B-v0.2

- Extended vocabulary to 32768
- Supports v3 Tokenizer
- Supports function calling

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

### Smaug-Llama-3-70B-Instruct

The model outperforms Llama-3-70B-Instruct substantially, and is on par with GPT-4-Turbo, on MT-Bench (see below).

https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct

# Embedding Models

### chandar-lab/NeoBERT

https://huggingface.co/chandar-lab/NeoBERT

NeoBERT is a next-generation encoder model for English text representation, pre-trained from scratch on the RefinedWeb dataset. NeoBERT integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an optimal depth-to-width ratio, and leverages an extended context length of 4,096 tokens. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves state-of-the-art results on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions.

### nomic-ai/nomic-embed-text-v2-moe

https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe

`nomic-embed-text-v2-moe` is SoTA multilingual MoE text embedding model that excels at multilingual retrieval:

- **High Performance**: SoTA Multilingual performance compared to ~300M parameter models, competitive with models 2x in size
- **Multilinguality**: Supports ~100 languages and trained on over 1.6B pairs
- **Flexible Embedding Dimension**: Trained with [**Matryoshka Embeddings**](https://arxiv.org/abs/2205.13147) with 3x reductions in storage cost with minimal performance degradations
- **Fully Open-Source**: Model weights, [**code**](https://github.com/nomic-ai/contrastors), and training data (see code repo) released

| **Model** | **Params (M)** | **Emb Dim** | **BEIR** | **MIRACL** | **Pretrain Data** | **Finetune Data** | **Code** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Nomic Embed v2** | 305 | 768 | 52.86 | **65.80** | ✅ | ✅ | ✅ |
| mE5 Base | 278 | 768 | 48.88 | 62.30 | ❌ | ❌ | ❌ |
| mGTE Base | 305 | 768 | 51.10 | 63.40 | ❌ | ❌ | ❌ |
| Arctic Embed v2 Base | 305 | 768 | **55.40** | 59.90 | ❌ | ❌ | ❌ |
|  |  |  |  |  |  |  |  |
| BGE M3 | 568 | 1024 | 48.80 | **69.20** | ❌ | ✅ | ❌ |
| Arctic Embed v2 Large | 568 | 1024 | **55.65** | 66.00 | ❌ | ❌ | ❌ |
| mE5 Large | 560 | 1024 | 51.40 | 66.50 | ❌ | ❌ | ❌ |

### Alibaba-NLP/gte-modernbert-base

https://huggingface.co/Alibaba-NLP/gte-modernbert-base

We are excited to introduce the gte-modernbert series of models, which are built upon the latest modernBERT pre-trained encoder-only foundation models. The gte-modernbert series models include both text embedding models and rerank models.

The gte-modernbert models demonstrates competitive performance in several text embedding and text retrieval evaluation tasks when compared to similar-scale models from the current open-source community. This includes assessments such as MTEB, LoCO, and COIR evaluation.

### llamaindex/vdr-2b-multi-v1

https://huggingface.co/llamaindex/vdr-2b-multi-v1

vdr-2b-multi-v1 is a multilingual embedding model designed for visual document retrieval across multiple languages and domains. It encodes document page screenshots into dense single-vector representations, this will effectively allow to search and query visually rich multilingual documents without the need for any OCR, data extraction pipelines, chunking...

- **Trained on 🇮🇹 Italian, 🇪🇸 Spanish, 🇬🇧 English, 🇫🇷 French and 🇩🇪 German:** together they form a new large, open-source, multilingual training dataset of 500k high-quality samples.
- **Cross-lingual Retrieval**: substantially better on real-world scenarios. For example, this allows for searching german documents with italian queries.
- **Matryoshka Representation Learning**: You can reduce the vectors size 3x and still keep 98% of the embeddings quality.

### google/Gemma-Embeddings-v1.0

https://huggingface.co/google/Gemma-Embeddings-v1.0

GemmaEmbed is a dense-vector embedding model, trained especially for retrieval. As of December 12, 2024, GemmaEmbed achieves the #1 position overall on the MTEB leaderboard, with a score of 72.72.

### Massive Text Embedding Benchmark

https://huggingface.co/spaces/mteb/leaderboard

### jinaai/jina-clip-v2

https://huggingface.co/jinaai/jina-clip-v2

jina-clip-v2 is a general-purpose multilingual multimodal embedding model for text & images.

Multimodal embeddings enable searching and understanding data across different modalities through a coherent representation. They serve as the backbone of neural information retrieval and multimodal GenAI applications.

### jxm/cde-small-v1

https://huggingface.co/jxm/cde-small-v1

Our new model that naturally integrates "context tokens" into the embedding process. As of October 1st, 2024, cde-small-v1 is the best small model (under 400M params) on the MTEB leaderboard for text embedding models, with an average score of 65.00.

### openbmb/MiniCPM-Embedding

https://huggingface.co/openbmb/MiniCPM-Embedding

MiniCPM-Embedding is trained based on MiniCPM-2B-sft-bf16 and incorporates bidirectional attention and Weighted Mean Pooling(Muennighoff, N. (2022). Sgpt: Gpt sentence embeddings for semantic search. arXiv preprint arXiv:2202.08904) in its architecture. The model underwent multi-stage training using approximately 6 million training examples, including open-source, synthetic, and proprietary data.

### dleemiller/word-llama-l2-supercat

https://huggingface.co/dleemiller/word-llama-l2-supercat

WordLlama is a fast, lightweight NLP toolkit that handles tasks like fuzzy-deduplication, similarity and ranking with minimal inference-time dependencies and optimized for CPU hardware.

### jinaai/jina-embeddings-v3

https://huggingface.co/jinaai/jina-embeddings-v3

jina-embeddings-v3 is a **multilingual multi-task text embedding model** designed for a variety of NLP applications. Based on the [**Jina-XLM-RoBERTa architecture**](https://huggingface.co/jinaai/xlm-roberta-flash-implementation), this model supports Rotary Position Embeddings to handle long input sequences up to **8192 tokens**. Additionally, it features 5 LoRA adapters to generate task-specific embeddings efficiently.

### answerdotai/answerai-colbert-small-v1

https://huggingface.co/answerdotai/answerai-colbert-small-v1

**answerai-colbert-small-v1** is a new, proof-of-concept model by [**Answer.AI**](https://answer.ai/), showing the strong performance multi-vector models with the new [**JaColBERTv2.5 training recipe**](https://arxiv.org/abs/2407.20750) and some extra tweaks can reach, even with just **33 million parameters**.

# Other Traditional Transformer Language Models

### iiiorg/piiranha-v1-detect-personal-information

[https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information?text=送孟浩然之广陵](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information?text=%E9%80%81%E5%AD%9F%E6%B5%A9%E7%84%B6%E4%B9%8B%E5%B9%BF%E9%99%B5)

Piiranha (cc-by-nc-nd-4.0 license) is trained to detect 17 types of Personally Identifiable Information (PII) across six languages. It successfully catches 98.27% of PII tokens, with an overall classification accuracy of 99.44%. Piiranha is especially accurate at detecting passwords, emails (100%), phone numbers, and usernames.

### jinaai/reader-lm-0.5b

https://huggingface.co/jinaai/reader-lm-0.5b

Jina Reader-LM is a series of models that convert HTML content to Markdown content, which is useful for content conversion tasks. The model is trained on a curated collection of HTML content and its corresponding Markdown content.

Hongsu notes: On Colab A100, it takes 7 minutes to convert 1,130 words of HTML to 1,094 words of Markdown.

# Other Models

### Mozilla/whisperfile

https://huggingface.co/Mozilla/whisperfile

Whisperfile is a high-performance implementation of OpenAI's Whisper created by Mozilla Ocho as part of the llamafile project, based on the whisper.cpp software written by Georgi Gerganov, et al.

### Qwen/Qwen2-Audio-7B-Instruct

https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

Qwen2-Audio is the new series of Qwen large audio-language models. Qwen2-Audio is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions. We introduce two distinct audio interaction modes:

voice chat: users can freely engage in voice interactions with Qwen2-Audio without text input;

audio analysis: users could provide audio and text instructions for analysis during the interaction;