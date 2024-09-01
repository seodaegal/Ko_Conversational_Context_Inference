import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import ORPOTrainer, ORPOConfig
from src.orpo_data import CustomDataset, custom_to_hf_dataset

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_id)

    #error handling for Qwen 
    if "Qwen" in args.model_id:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


    train_dataset = CustomDataset("resource/data/data_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/data_dev.json", tokenizer)

    train_dataset_hf = custom_to_hf_dataset(train_dataset)
    valid_dataset_hf = custom_to_hf_dataset(valid_dataset)

    training_args = ORPOConfig(
        output_dir=args.save_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=True,
        logging_dir='./logs',
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
    )

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_hf,
        eval_dataset=valid_dataset_hf,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train", description="Training for Conversational Context Inference using ORPO.")
    
    g = parser.add_argument_group("Common Parameters")
    g.add_argument("-m","--model_id", type=str, required=True, help="Model file path")
    g.add_argument("--tokenizer", type=str, help="HuggingFace tokenizer path")
    g.add_argument("--save_dir", required=True, type=str, default="resource/results", help="Model save path")
    g.add_argument("--batch_size", type=int, default=1, help="Batch size (both train and eval)")
    g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    g.add_argument("--warmup_steps", type=int, default=1, help="Scheduler warmup steps")
    g.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    g.add_argument("--epoch", type=int, default=1, help="Training epochs")
    #g.add_argument("--report_to", type=str, default=None, help="Reporting integration (e.g., wandb)"))
    #g.add_argument("--bf16", action="store_true", help="Use bf16 precision")

    args = parser.parse_args()
    main(args)
