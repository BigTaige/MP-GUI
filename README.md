
## MP-GUI: Modality Perception with MLLMs for GUI Understanding (CVPR 2025)
## Setup

1. **Configure the Runtime Environment**

   Execute the following script to set up the necessary environment:

   ```bash
   sh env.sh
   ```

2. **Download Intern2-VL-8B**
Intern2-VL-8B will be used to initialize the visual backbone, alignment projector and LLM. The other modules of MP-GUI are randomly initialized and trained from scratch as follows:
   [https://modelscope.cn/models/OpenGVLab/InternVL2-8B]

## Multi-step Training

Follow the steps below to complete the multi-step training process:

1. **Training Textual Perceiver**

   ```bash
   sh MP-GUI/model/shell/multi_step_training/train_textual_perceiver.sh
   ```

2. **Training Graphical Perceiver**

   ```bash
   sh MP-GUI/model/shell/multi_step_training/train_graphical_perceiver.sh
   ```

3. **Training Spatial Perceiver**

   ```bash
   sh MP-GUI/model/shell/multi_step_training/train_spatial_perceiver.sh
   ```

4. **Training Fusion Gate**

   ```bash
   sh MP-GUI/model/shell/multi_step_training/train_fusion_gate.sh
   ```

5. **Complete Training on Benchmark**

   ```bash
   sh MP-GUI/model/shell/multi_step_training/benchmark.sh
   ```

## Datasets

The following open-source datasets are used in this project:

- **AMEX:** [here](https://yuxiangchai.github.io/AMEX/)
- **AITW:** [here](https://github.com/google-research/google-research/tree/master/android_in_the_wild)
- **Rico:** [here](http://www.interactionmining.org/rico.html#quick-downloads)
- **Semantic UI:** [here](http://www.interactionmining.org/rico.html)


### Dataset Preparation

1. **Download the Datasets**

   Download each dataset and place them in the `MP-GUI/training_data` directory.

2. **Update Image Paths**

   Replace the `image_path` entries in the datasets with your local image paths.

3. **AITW Specific Processing**

   For [AITW](https://github.com/google-research/google-research/tree/master/android_in_the_wild), we randomly sample from the **"general"** and **"install"** categories and process them using the following script:

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
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-72B-Instruct --model Qwen/Qwen2-VL-72B-Instruct -tp 8
```
```bash
python pipeline/vllm_pipeline_v2.py
```

Additionally, the `MP-GUI/data_tools` directory provides scripts to create spatial relationship prediction data based on the [Semantic UI dataset](http://www.interactionmining.org/rico.html).

```bibtex
@misc{wang2025mpguimodalityperceptionmllms,
      title={MP-GUI: Modality Perception with MLLMs for GUI Understanding}, 
      author={Ziwei Wang and Weizhi Chen and Leyang Yang and Sheng Zhou and Shengchu Zhao and Hanbei Zhan and Jiongchao Jin and Liangcheng Li and Zirui Shao and Jiajun Bu},
      year={2025},
      eprint={2503.14021},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.14021}, 
}

