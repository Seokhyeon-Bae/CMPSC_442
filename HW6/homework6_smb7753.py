#!/usr/bin/env python3

import os
import torch
import json
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb

# ===============================
# SETUP (CPU version)
# ===============================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.backends.mps.is_available = lambda: False
device = torch.device("cpu")
print(f"Using device: {device}")

# Save device info to log file
log_path = "example_log.txt"
with open(log_path, "a") as f:
    f.write(f"Using device: {device}\n")

# ====================
# WANDB Login (Optional)
# ====================
try:
    wandb.login()
    wandb_enabled = True
    print("Wandb login successful")
except:
    wandb_enabled = False
    print("Wandb login failed, proceeding without wandb")

# ===============================
# Load Dataset
# ===============================
dataset = load_dataset("mlabonne/smoltldr")
print("Dataset loaded successfully!")
print(dataset)

# Save dataset info and print 3 examples
with open(log_path, "a") as f:
    f.write("\nDataset structure:\n")
    f.write(str(dataset) + "\n")
for i in range(3):
    prompt = dataset["train"][i]["prompt"]
    completion = dataset["train"][i]["completion"]
    print(f"\nExample {i+1}")
    print("Prompt:", prompt)
    print("Completion:", completion)
    with open(log_path, "a") as f:
        f.write(f"\nExample {i+1}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Completion: {completion}\n")

# ===============================
# Hyperparameters
# ===============================
learning_rate = 5e-5
per_device_train_batch_size = 8
gradient_accumulation_steps = 1
max_prompt_length = 256
max_completion_length = 64
num_generations = 1
num_train_epoch = 1

# ===============================
# Reward Function
# ===============================
def reward_len(completions, **kwargs):
    """
    Reward function that encourages completions of length 50.
    Returns a list of float rewards (one per completion).
    """
    target_len = 50
    rewards = []
    for completion in completions:
        length = len(completion.split())
        reward = 1.0 - abs(length - target_len) / target_len
        reward = max(0.01, reward)
        rewards.append(reward)
    return rewards


# ===============================
# Train GRPO Model
# ===============================
def train_model(dataset):
    from transformers import LogitsProcessorList, InfNanRemoveLogitsProcessor

    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Patch safe generate
    original_generate = model.generate

    def safe_generate(*args, **kwargs):
        if "logits_processor" not in kwargs or kwargs["logits_processor"] is None:
            kwargs["logits_processor"] = LogitsProcessorList()
        kwargs["logits_processor"].append(InfNanRemoveLogitsProcessor())
        kwargs.setdefault("temperature", 0.7)
        kwargs.setdefault("top_p", 0.9)
        kwargs.setdefault("max_new_tokens", 64)
        return original_generate(*args, **kwargs)

    model.generate = safe_generate

    # GRPO config
    training_args = GRPOConfig(
        output_dir="./GRPO_mac",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        optim="adamw_torch",
        num_train_epochs=num_train_epoch,
        fp16=False,
        bf16=False,
        report_to=["wandb"] if wandb_enabled else [],
        remove_unused_columns=False,
        logging_steps=10,
        run_name="SmolGRPO-CPU-Mac",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_len],
        args=training_args,
        train_dataset=dataset["train"].select(range(50)),
    )

    if wandb_enabled:
        wandb.init(project="SmolGRPO-Mac")

    trainer.train()

    if wandb_enabled:
        wandb.finish()
        
    # Save model
    output_dir = "./GRPO_mac/final_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer, trainer

# save log
def save_log(model, tokenizer, trainer, dataset, test_prompt, test_summary, eval_dataset):
    log_path = "homework6_log.txt"
    with open(log_path, "w") as f:
        # Device Info
        f.write(f"device: torch.device = torch.device(torch._C._get_default_device())\n")
        f.write(f"{device}\n")
        f.write(f"{device.type}\n\n")

        # Dataset
        f.write(str(dataset) + "\n\n")

        # Parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params
        f.write(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_percent:.4f}\n\n")

        # Training logs
        if trainer.state.log_history:
            for log in trainer.state.log_history:
                f.write(json.dumps(log) + "\n\n")

        # Model Saved
        f.write("Model saved to ./GRPO_mac/final_model\n\n")

        # Test prompt + TL;DR
        f.write("Test prompt:\n")
        f.write(test_prompt.strip() + "\n\n")
        f.write(f"TL;DR: {test_summary}\n\n")

        # Evaluation
        f.write(f"Generating {len(eval_dataset)} examples for evaluation...\n\n")

        for i, example in enumerate(eval_dataset):
            prompt = example["prompt"]
            reference = example["completion"]
            summary = run_inference(prompt, model, tokenizer)

            f.write(f"Processing example {i+1}/{len(eval_dataset)}\n")
            f.write(f"Generated: {summary}\n")
            f.write(f"Reference: {reference}\n\n")

# ===============================
# Inference
# ===============================
def run_inference(prompt: str, model=None, tokenizer=None) -> str:
    if model is None or tokenizer is None:
        model_path = "./GRPO_mac/final_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad(): 
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        if torch.isnan(output_ids).any():
            print("Warning: NaN detected in output. Skipping this sample.")
            return "Generation Error"

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary = generated_text[len(prompt):].strip()
    if summary.startswith("TL;DR:"):
        summary = summary[7:].strip()

    return summary


# ===============================
# Generate Evaluation Set
# ===============================
def generate_evaluations(model, tokenizer, dataset, num_examples=50):
    output_file = "evaluation_results.json"

    print(f"\nGenerating {num_examples} examples for evaluation...")
    results = []

    eval_examples = dataset["test"].select(range(min(num_examples, len(dataset["test"]))))

    for i, example in enumerate(eval_examples):
        prompt = example["prompt"]
        reference = example["completion"]

        print(f"\nProcessing example {i + 1}/{num_examples}")
        summary = run_inference(prompt, model, tokenizer)

        print(f"Generated: {summary}")
        print(f"Reference: {reference}")

        results.append({
            "id": i,
            "prompt": prompt,
            "generated": summary,
            "reference": reference,
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to {output_file}")
    return results

# ===============================
# 🔍 Main
# ===============================
if __name__ == "__main__":
    # Train
    model, tokenizer, trainer = train_model(dataset)

    # Test Inference
    test_prompt = """
    The cat (Felis catus), also referred to as the domestic cat or house cat, is a small
    domesticated carnivorous mammal. It is the only domesticated species of the family Felidae.
    """
    
    test_summary = run_inference(test_prompt, model, tokenizer)
    eval_dataset = dataset["test"].select(range(50))
    save_log(model, tokenizer, trainer, dataset, test_prompt, test_summary, eval_dataset)

    # Generate evaluation results
    generate_evaluations(model, tokenizer, dataset, num_examples=50)