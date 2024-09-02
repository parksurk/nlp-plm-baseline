import os
import yaml
import torch
import wandb
import argparse
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from chat_summarization.dataset import Preprocess, prepare_train_dataset, compute_metrics

def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    bart_config = BartConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], config=bart_config)

    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)

    return generate_model, tokenizer

def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],  # 모델 체크포인트 저장 경로
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        report_to=config['training']['report_to']
    )

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        config=config
    )

    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"] = "false"

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[early_stopping_callback]
    )

    return trainer

def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)

    # Prepare dataset
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    # Load trainer
    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for PLM summarization.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
