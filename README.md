
<div align="center">
  <h3 align="center">Fine-tuning LLMs on Custom Datasets</h3>
  <p align="center">
    Example project for fine-tuning, running inference, and evaluating LLMs using <a href="https://pytorch.org/torchtune/">TorchTune</a> on <a href="https://uwconnect.uw.edu/it?id=kb_article_view&sysparm_article=KB0036077">UW's Tillicum</a>.
    <br />
    <br />
    <a href="#getting-started">Getting Started</a>
    ·
    <a href="#usage">Usage</a>
    ·
    <a href="#additional-resources">Additional Resources</a>
  </p>
</div>

## Getting Started

This project uses [TorchTune](https://pytorch.org/torchtune/) to demonstrate how to fine-tune LLMs with LoRA on [Tillicum's](https://uwconnect.uw.edu/it?id=kb_article_view&sysparm_article=KB0036077) multi-GPU nodes.

### Prerequisites

* Hugging Face account with [Llama 3.2 access](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
* [SSH access](https://hyak.uw.edu/docs/tillicum/) to Tillicum.

### Installation

1. SSH into Tillicum and connect to a GPU instance: 
   ```sh
   salloc --gpus=1
   ```

2. Clone the repository:
   ```sh
   git clone https://github.com/josecols/ft-llms-tillicum.git ft-llms
   cd ft-llms
   ```
   
3. Set up the project path environment variable:
   
   Add the following line to your `~/.bashrc` file (replace the path with your actual project location):
   ```sh
   export FT_LLMS_ROOT=/gpfs/scrubbed/<netid>/projects/ft-llms
   ```
   
   Then reload your bash configuration:
   ```sh
   source ~/.bashrc
   ```
 
4. Enable the conda module: 
   ```sh
   module load conda
   ```

5. Create and activate conda environment:
   ```sh
   conda create -n ft-llms python=3.12
   conda activate ft-llms
   ```

6. Install TorchTune dependencies:
   ```sh
   pip install torch torchvision torchao
   ```

7. Install TorchTune:
   ```sh
   pip install torchtune
   ```

8. Install [WanDB](https://wandb.ai/site/) to track fine-tuning jobs:
   ```sh
   pip install wandb
   ```
   > **Note:** You will need to [authenticate](https://docs.wandb.ai/models/quickstart#install-the-wandb-library-and-log-in) your WanDB account for the first time.

9. Install HuggingFace libraries to run **inference tasks**:
   ```sh
   pip install transformers peft accelerate
   ```

10. Install the ROUGE score package to run **evaluation tasks**:
    ```sh
    pip install rouge-score
    ```
   
11. Download the NLTK data (required by the ROUGE package):
    ```sh
    python -c "import nltk; nltk.download('punkt_tab')"
    ```

## Usage

### Prepare the Dataset

This workshop demo uses the [PLOS dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-PLOS) from [BioLaySumm 2025](https://biolaysumm.org) to fine-tune a model for lay summarization of biomedical articles.

To download and prepare the dataset for training, run the following command:

```sh
python scripts/prepare_dataset.py --use-abstracts
```

> [!NOTE]
> You can omit the `--use-abstracts` flag if you prefer to train with the full article texts as input. However, you might need to adjust the training configuration to prevent out-of-memory errors.

### Download the Model

The demo uses the Llama 3.2 3B model, but you can choose a different model if you prefer. To see all the supported model configurations, run `tune ls`.

```sh
tune download meta-llama/Llama-3.2-3B-Instruct --output-dir models/Llama-3.2-3B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf-token <your-token>
```

> [!NOTE]
> Make sure to replace `<your-token>` with your HuggingFace token. You can also set it via the `HF_TOKEN` [environment variable](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken)

### Fine-Tuning

Run single-node fine-tuning on 8 GPUs:

```sh
sbatch tasks/train_8_gpus.slurm
```

There is also a multi-node example script (`tasks/train_16_gpus.slurm`) that you can adapt for various distributed setups.

To check the job's progress, use the `squeue -u <netid>` command.

### Inference

Run the following command to generate the summaries with the fine-tuned model:

```sh
sbatch tasks/inference.slurm
```

### Evaluation

Run the following command to evaluate the model summaries against the gold-standard:

```sh
sbatch tasks/eval.slurm
```

## Additional Resources

- [Tillicum Documentation](https://hyak.uw.edu/docs/tillicum/)
- [Research Computing at the UW](https://it.uw.edu/guides/research/)
- [Research Computing Club](https://depts.washington.edu/uwrcc/)
- [TorchTune Documentation](https://meta-pytorch.org/torchtune/main/)