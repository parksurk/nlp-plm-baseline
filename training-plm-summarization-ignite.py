import os
import yaml
import torch
import wandb
import argparse
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from torch.utils.data import DataLoader
from ignite.metrics import RunningAverage
from chat_summarization.dataset import Preprocess, prepare_train_dataset, compute_metrics
from datasets import load_metric

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
        output_dir=config['general']['output_dir'],
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

    train_loader = DataLoader(train_inputs_dataset, batch_size=config['training']['per_device_train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_inputs_dataset, batch_size=config['training']['per_device_eval_batch_size'])

    # Load trainer
    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_loader, val_loader)

    # Ignite engine for training and evaluation
    def update_engine(engine, batch):
        return trainer.prediction_step(trainer.model, batch, prediction_loss_only=False)

    def evaluation_step(engine, batch):
        return trainer.prediction_step(trainer.model, batch, prediction_loss_only=True)

    ignite_trainer = Engine(update_engine)
    ignite_evaluator = Engine(evaluation_step)

    # Modify RunningAverage to correctly extract the loss from the tuple
    RunningAverage(output_transform=lambda x: x[0]).attach(ignite_trainer, 'training_loss')
    RunningAverage(output_transform=lambda x: x[0]).attach(ignite_evaluator, 'validation_loss')

    # Initialize the best validation loss
    global best_val_loss
    best_val_loss = float('inf')

    @ignite_trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        global best_val_loss  # Ensure that best_val_loss is recognized as a global variable
        epoch = engine.state.epoch
        wandb.log({'epoch': epoch, 'training_loss': engine.state.metrics['training_loss']})

        # Evaluate the model and save checkpoint if the performance improves
        ignite_evaluator.run(val_loader)
        val_loss = ignite_evaluator.state.metrics['validation_loss']
        wandb.log({'epoch': engine.state.epoch, 'validation_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(output_dir=os.path.join(config['general']['output_dir'], f"checkpoint-epoch-{epoch}"))

    # Monitor learning rate
    @ignite_trainer.on(Events.ITERATION_COMPLETED)
    def log_learning_rate(engine):
        if trainer.optimizer is not None:  # Check if the optimizer is initialized
            lr = trainer.optimizer.param_groups[0]['lr']
            wandb.log({'learning_rate': lr, 'iteration': engine.state.iteration})

    # Monitor gradients
    @ignite_trainer.on(Events.ITERATION_COMPLETED)
    def log_gradients(engine):
        for name, param in generate_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                wandb.log({f'gradients/{name}': torch.norm(param.grad).item(), 'iteration': engine.state.iteration})

    # Load and attach the ROUGE metric
    rouge = load_metric("rouge")

    @ignite_evaluator.on(Events.COMPLETED)
    def compute_rouge(engine):
        # Ensure engine.state.output is not None
        if engine.state.output is not None:
            predictions = engine.state.output[1]
            references = engine.state.output[2]
            
            if predictions is not None and references is not None:
                decoded_preds = [trainer.tokenizer.decode(g, skip_special_tokens=True) for g in predictions]
                decoded_refs = [trainer.tokenizer.decode(l, skip_special_tokens=True) for l in references]
                
                rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_refs)
                wandb.log({'rouge1': rouge_scores['rouge1'].mid.fmeasure,
                           'rouge2': rouge_scores['rouge2'].mid.fmeasure,
                           'rougeL': rouge_scores['rougeL'].mid.fmeasure,
                           'epoch': engine.state.epoch})

    pbar = ProgressBar()
    pbar.attach(ignite_trainer)
    
    # Run the trainer
    ignite_trainer.run(train_loader, max_epochs=config['training']['num_train_epochs'])

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for PLM summarization.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
