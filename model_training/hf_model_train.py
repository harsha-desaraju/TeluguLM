from transformers import PreTrainedTokenizerFast
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from transformers import Trainer
from transformers import TrainingArguments
import os, shutil




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

    config = GPT2Config(
        vocab_size=16384,
        n_positions=512,
        n_ctx=512,
        n_embd=512,
        n_layer=8,
        n_head=8,
    )

    model = GPT2LMHeadModel(config)

    total_params = 0
    for layer in model.parameters():
        total_params += layer.numel()
    print(f"Total parameters of the model are: {total_params}")


    dataset = load_from_disk("../dataset_creation/dataset")

    train_dataset = dataset['train'].select(range(1000))
    test_dataset = dataset['test'].select(range(10))

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'], num_proc=10)
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['text'], num_proc=10)

    # tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=10)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT = causal LM
    )

    training_args = TrainingArguments(
        output_dir="./telugu-gpt",
        # overwrite_output_dir=True,

        # --- Training Duration ---
        num_train_epochs=5,

        # --- Batch Size & Accumulation ---
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,  # Effective batch = 32 * 4 = 128

        # --- Optimizer & Scheduler ---
        learning_rate=3e-4,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        max_grad_norm=1.0,

        # --- Precision & Performance ---
        fp16=True,
        dataloader_num_workers=2,
        gradient_checkpointing=False,

        # --- Evaluation & Saving ---
        eval_strategy="steps",
        eval_steps=100,
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
        compute_metrics=compute_metrics
    )

    # --- Resume if checkpoint exists ---
    checkpoint_path = "./telugu-gpt"
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
    trainer.save_model("./telugu-gpt/final-checkpoint")
    tokenizer.save_pretrained("./telugu-gpt/final-checkpoint")

    print("Training complete for this session.")