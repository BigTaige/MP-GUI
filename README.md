```markdown
# Model Startup Guide

Welcome to the **Intern2-VL-8B** model repository. This guide will walk you through setting up the environment, training the model, and preparing the necessary datasets.

## Prerequisites

- Ensure you have `sh` installed on your system.
- Git installed for repository cloning.
- Access to the required datasets (links provided below).

## Setup

1. **Configure the Runtime Environment**

   Execute the following script to set up the necessary environment:

   ```bash
   sh env.sh
   ```

2. **Download Intern2-VL-8B**

   [Provide the link or instructions to download Intern2-VL-8B here.]

## Multi-step Training

Follow the steps below to complete the multi-step training process:

1. **Train Textual Perceiver**

   ```bash
   MP-GUI/model/shell/multi_step_training/train_textual_perceiver.sh
   ```

2. **Train Graphical Perceiver**

   ```bash
   MP-GUI/model/shell/multi_step_training/train_graphical_perceiver.sh
   ```

3. **Train Spatial Perceiver**

   ```bash
   MP-GUI/model/shell/multi_step_training/train_spatial_perceiver.sh
   ```

4. **Train Fusion Gate**

   ```bash
   MP-GUI/model/shell/multi_step_training/train_fusion_gate.sh
   ```

5. **Complete Training on Benchmark**

   ```bash
   MP-GUI/model/shell/multi_step_training/benchmark.sh
   ```

## Datasets

The following open-source datasets are used in this project:

- **AMEX:** [http://xxx](http://xxx)
- **AITW:** [http://xxx](http://xxx)
- **Rico:** [http://xxx](http://xxx)
- **Semantic UI:** [http://xxx](http://xxx)

### Dataset Preparation

1. **Download the Datasets**

   Download each dataset and place them in the `MP-GUI/training_data` directory.

2. **Update Image Paths**

   Replace the `image_path` entries in the datasets with your local image paths.

3. **AITW Specific Processing**

   For AITW, we randomly sample from the **general** and **install** categories and process them using the following script:

   ```bash
   python data_tools/get_small_icon_grounding_data.py
   ```

## Post-training Steps

After completing each training step, merge the LoRA parameters and update the weight paths in the training scripts:

1. **Merge LoRA Parameters**

   ```bash
   python MP-GUI/model/tools/merge_lora.py
   ```

2. **Update Weight Paths**

   Replace the corresponding weight paths in your training scripts with the merged weights.

## Synthetic Data Generation

You can generate a vast amount of synthetic data using the Qwen2-VL-72B model within the vLLM architecture:

```bash
python pipeline/vllm_pipeline_v2.py
```

Additionally, the `MP-GUI/data_tools` directory provides scripts to create spatial relationship prediction data based on the Semantic UI dataset.

