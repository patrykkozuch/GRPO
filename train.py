# train_grpo.py
import wandb

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from rewards import *

dataset = load_from_disk("dataset")

training_args = GRPOConfig(
    max_steps=10_000,
    output_dir="Qwen3-0.6B-GRPO-COOK-Stable",
    learning_rate=1e-6,
    logging_steps=10,
    num_generations=8,
    max_completion_length=512,
    report_to="wandb",
    use_vllm=True,
    log_completions=True,
    num_completions_to_print=3,
    beta=0.005,
    gradient_accumulation_steps=8,
    save_steps=50
)

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="patryk-kozuch-personal",
    # Set the wandb project where this run will be logged.
    project="GRPO"
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[
        preparation_time_reward,
        portions_reward,
        ingredients_structure_reward,
        steps_structure_reward,
        tag_order_reward,
        has_finished_thinking
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
