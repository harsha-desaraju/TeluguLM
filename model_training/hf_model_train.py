import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from transformers import PreTrainedTokenizerFast
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer
from transformers import TrainingArguments


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    import torch
    import torch.nn.functional as F

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    perplexity = torch.exp(loss)

    return {"perplexity": perplexity.item()}



def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


if __name__ == '__main__':

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="../tokenization/telugu_tokenizer/telugu_unigram_tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    config = LlamaConfig(
        vocab_size=16384,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act='silu',
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    model = LlamaForCausalLM(config)

    total_params = 0
    for layer in model.parameters():
        total_params += layer.numel()
    print(f"Total parameters of the model are: {total_params}")

    dataset = load_dataset("harsha-desaraju/telugu-text")
    train_dataset = dataset['train']
    test_dataset = dataset['test'].select(range(1000))

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'], num_proc=4)
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['text'], num_proc=4)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT = causal LM
    )

    training_args = TrainingArguments(
        output_dir="./telugu-llama",
        # overwrite_output_dir=True,

        # --- Training Duration ---
        num_train_epochs=2,

        # --- Batch Size & Accumulation ---
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,  # Effective batch = 16 * 4 = 64

        # --- Optimizer & Scheduler ---
        learning_rate=1e-4,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,

        # --- Precision & Performance ---
        fp16=True,
        dataloader_num_workers=2,
        gradient_checkpointing=False,

        # --- Evaluation & Saving ---
        eval_strategy="steps",
        eval_steps=100,
        eval_accumulation_steps=8,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # --- Logging ---
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
    )



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics
    )

    # --- Resume if checkpoint exists ---
    checkpoint_path = "./telugu-llama"
    last_checkpoint = None

    if os.path.isdir(checkpoint_path):
        checkpoints = [
            os.path.join(checkpoint_path, d)
            for d in os.listdir(checkpoint_path)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from: {last_checkpoint}")
        else:
            print("No checkpoint found. Starting fresh.")


    # --- Train (resumes automatically if checkpoint found) ---
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # --- Save final model + tokenizer ---
    trainer.save_model(f"{checkpoint_path}/final-checkpoint")
    tokenizer.save_pretrained(f"{checkpoint_path}/final-checkpoint")

    print("Training complete for this session.")