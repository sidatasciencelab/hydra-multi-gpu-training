# Multi GPU Training

Distributing training across multiple GPUs can be challenging to do manually. If you are using Transformers and PyTorch, the easiest way to train on multiple GPUs is via `torchrun`. For this example, we will be fine-tuning Whisper, an audio-speech recognition model from OpenAI. We will fine-tune it to an unsupported language, Xhosa, which is spoken in Southern Africa and is a member of the Bantu languages. To follow along, you will not need to know anything about ASR or Xhosa.

## What is Torchrun?

Torchrun is a tool for running PyTorch scripts on multiple GPUs, making your deep learning tasks faster and more efficient. Think of it as a helper that takes your regular PyTorch program and spreads the work across several GPUs, allowing them to work together on the same task. This is especially useful for training large models or working with big datasets that would take too long on a single GPU. Torchrun handles a lot of the complex setup required for distributed computing, so you don't have to worry about the low-level details. While it may seem a bit advanced at first, torchrun can significantly speed up your deep learning workflows once you get the hang of it.

## Example

Let's take a look at how we can leverage torchrun in our workflow.

### Job File

First, we will want to specify that we are using GPUs. To do this in our job file, we will add the following line to our parameters:

```bash
#$ -l gpu,gpu_arch=L40S
```

This tells Hydra to load a GPU node, specifically one with the L40S. Currently, this node has 4 L40S. When working with the GPU node, you will be limited to the number of threads you can specify. The current max is 4. You can use this to call up extra virtual memory. Here's an example of this line in the job file.

```bash
#$ -pe mthread 4
```

Our entire parameters for a sample project will look something like this:

```bash
# /bin/sh
# ----------------Parameters---------------------- #
#$ -S /bin/sh
#$ -pe mthread 4
#$ -l gpu,gpu_arch=L40S
#$ -cwd
#$ -j y
#$ -N whisper-zulu-medium
#$ -o whisper-zulu-medium.log
#$ -m bea
#$ -M mattinglyw@si.edu
```

Next, we need to specify the modules that we will be running. For this project, I'm using a nearly identical to environment to another project that worked with DNA.

In order to leverage the GPUs, we need to make sure that we load the `nvidia` module.

```bash
module load nvidia
```

Next, we need to add the directory of my conda environment to the beginning of the PATH environment variable. It ensures that commands are first looked for in the dna-gpu conda environment.

```bash
export PATH=/home/mattinglyw/mambaforge/envs/dna-gpu/bin:$PATH
```

Next, we set the XLA_FLAGS environment variable, specifying the CUDA data directory for XLA GPU operations.

```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/mattinglyw/mambaforge/envs/dna-gpu/lib
```

Finally, we add a directory to the LD_LIBRARY_PATH. This helps the system find necessary shared libraries, specifically for NVIDIA's cuDNN in this case.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/bioinformatics/dna-algorithm/mamba/envs/dna-algorithm/lib/python3.9/site-packages/nvidia/cudnn/lib/
```

When we bring the module section of our job file together, it will look something like this.

```bash
# ----------------Modules------------------------- #
#
#module load bio/dna-algorithm/
module load nvidia
# Update PATH
export PATH=/home/mattinglyw/mambaforge/envs/dna-gpu/bin:$PATH
# Set XLA flags
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/mattinglyw/mambaforge/envs/dna-gpu/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/bioinformatics/dna-algorithm/mamba/envs/dna-algorithm/lib/python3.9/site-packages/nvidia/cudnn/lib/
#
```

Now that we have our parameters and modules setupm, we need to run our commands. Unlike a traditional Python file, we will not use the command `python [filename].py`. Instead, we will leverage `torchrun`.

Torch run will take an additional parameter: `nproc_per_node` which specifies the number of GPUs you want to leverage. Since we are working on a node with 4 GPUs, we will specify that we want 4 GPUs. After this, we will specify the Python file that we want `torchrun` to run. In our case: `training-seq2seq.py`. After this, all the following arguments will be those expected by the Python file.

```bash
torchrun \
 	--nproc_per_node 4 training-seq2seq.py \
	--model_name_or_path="openai/whisper-medium" \
	--dataset_name="wjbmattingly/xhosa_merged_audio" \
	--language="swahili" \
	--task="transcribe" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--max_steps="5000" \
	--output_dir="./whisper-xhosa-medium" \
	--per_device_train_batch_size="16" \
	--per_device_eval_batch_size="16" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--eval_strategy="steps" \
	--eval_steps="500" \
	--save_strategy="steps" \
	--save_steps="500" \
	--generation_max_length="225" \
	--preprocessing_num_workers="16" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
    --push_to_hub \
    --hub_model_id "wjbmattingly/xhosa-medium" \
    --hub_private_repo \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate
```

### Python File

When we construct our Python file, we want to keep it as flexible as possible. In other words, we want a script that can be run with a single and multi-gpu support. A good example of this is the following [Python file from the Transformers](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py) library. For this project, I've called this file `training-seq2seq.py`.
