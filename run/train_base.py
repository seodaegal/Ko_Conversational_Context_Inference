
import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

from src.data import CustomDataset, DataCollatorForSupervisedDataset


# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, help="scheduler warmup steps")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=100, help="training epoch")
g.add_argument("--eval_steps", type=int, default=500, help="steps between evaluations")
g.add_argument("--early_stopping_patience", type=int, default=3, help="patience for early stopping")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id ,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.tokenizer == None:
        args.tokenizer = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CustomDataset("resource/data/data_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/data_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.1,
        num_train_epochs=args.epoch,
        max_steps=-1,
        torch_empty_cache_steps=100,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        log_level="info",
        logging_steps=1,
        logging_dir="/root/Ko_Conversational_Context_Inference/event_logs",
        save_strategy="epoch",
        save_total_limit=args.epoch,
        bf16=True,
        gradient_checkpointing=True,
        packing=True,
        seed=42,
        eval_steps=args.eval_steps,
        load_best_model_at_end = True,
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        callbacks=[early_stopping_callback],
    )

    trainer.train()


if __name__ == "__main__":
    exit(main(parser.parse_args()))
