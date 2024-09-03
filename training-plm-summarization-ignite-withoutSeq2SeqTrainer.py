import os
import yaml
import torch
import wandb
import argparse
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
from torch.utils.data import DataLoader
from ignite.metrics import RunningAverage
from datasets import load_metric
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

def create_trainer_and_evaluator(config, generate_model, tokenizer, optimizer, device):
    def update_engine(engine, batch):
        generate_model.train()
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = generate_model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    def evaluation_step(engine, batch):
        generate_model.eval()
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = generate_model(**inputs)
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            references = inputs['labels']
        return loss.item(), predictions, references

    trainer = Engine(update_engine)
    evaluator = Engine(evaluation_step)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'training_loss')
    RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'validation_loss')

    return trainer, evaluator

def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb before any logging
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        config=config
    )

    # Load tokenizer and model
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)

    # Prepare dataset
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    train_loader = DataLoader(train_inputs_dataset, batch_size=config['training']['per_device_train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_inputs_dataset, batch_size=config['training']['per_device_eval_batch_size'])

    # Initialize optimizer
    optimizer = torch.optim.AdamW(generate_model.parameters(), lr=config['training']['learning_rate'])

    # Create trainer and evaluator
    trainer, evaluator = create_trainer_and_evaluator(config, generate_model, tokenizer, optimizer, device)

    # Set up early stopping
    def score_function(engine):
        return -engine.state.metrics['validation_loss']

    early_stopping_handler = EarlyStopping(patience=config['training']['early_stopping_patience'], score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    # Progress bar
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['training_loss'])

    # Initialize the best validation loss
    global best_val_loss  # 전역 변수 선언
    best_val_loss = float('inf')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        global best_val_loss  # 전역 변수로 접근
        epoch = engine.state.epoch
        wandb.log({'epoch': epoch, 'training_loss': engine.state.metrics['training_loss']})

        # Run evaluation
        evaluator.run(val_loader)
        val_loss = evaluator.state.metrics['validation_loss']
        wandb.log({'epoch': epoch, 'validation_loss': val_loss})

        # Save checkpoint if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = os.path.join(config['general']['output_dir'], f"checkpoint-epoch-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            generate_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))

    # ROUGE metric computation
    rouge = load_metric("rouge")

    @evaluator.on(Events.COMPLETED)
    def compute_rouge(engine):
        predictions = engine.state.output[1]
        references = engine.state.output[2]

        if predictions is not None and references is not None:
            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in predictions]
            decoded_refs = [tokenizer.decode(l, skip_special_tokens=True) for l in references]

            rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_refs)
            wandb.log({'rouge1': rouge_scores['rouge1'].mid.fmeasure,
                       'rouge2': rouge_scores['rouge2'].mid.fmeasure,
                       'rougeL': rouge_scores['rougeL'].mid.fmeasure,
                       'epoch': engine.state.epoch})

    # Run the training
    trainer.run(train_loader, max_epochs=config['training']['num_train_epochs'])

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for PLM summarization.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
