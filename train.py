import argparse
import random
import yaml
import wandb  # Wandb를 import하여 사용

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from simple_ntc.bert_trainer import BertTrainer as Trainer
from simple_ntc.bert_dataset import TextClassificationDataset, TextClassificationCollator
from simple_ntc.utils import read_text


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_loaders(config, tokenizer):
    # Get list of labels and list of texts.
    labels, texts = read_text(config['train_fn'])

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - config['valid_ratio']))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], labels[:idx]),
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config['max_length']),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts[idx:], labels[idx:]),
        batch_size=config['batch_size'],
        collate_fn=TextClassificationCollator(tokenizer, config['max_length']),
    )

    return train_loader, valid_loader, index_to_label


def get_optimizer(model, config):
    if config['use_radam']:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config['lr'])
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config['lr'],
            eps=config['adam_epsilon']
        )

    return optimizer


def main(config):
    # Initialize Wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['wandb']['name'],
        config=config  # Track the full configuration
    )
    
    # Get pretrained tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(config['pretrained_model_name'])
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders(config, tokenizer)

    print(
        '|train| =', len(train_loader) * config['batch_size'],
        '|valid| =', len(valid_loader) * config['batch_size'],
    )

    n_total_iterations = len(train_loader) * config['n_epochs']
    n_warmup_steps = int(n_total_iterations * config['warmup_ratio'])
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model_loader = AlbertForSequenceClassification if config['use_albert'] else BertForSequenceClassification
    model = model_loader.from_pretrained(
        config['pretrained_model_name'],
        num_labels=len(index_to_label)
    )
    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config['gpu_id'] >= 0:
        model.cuda(config['gpu_id'])
        crit.cuda(config['gpu_id'])

    # Start train, passing the wandb object to Trainer
    trainer = Trainer(config, wandb)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    # Log model weights and artifacts to Wandb
    if config['wandb']['log_model']:
        wandb.watch(model, log="all")

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config['model_fn'])

    # Finish the Wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune PLM model with configuration file")
    parser.add_argument('--config_path', required=True, help="Path to the configuration file to load")
    
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
