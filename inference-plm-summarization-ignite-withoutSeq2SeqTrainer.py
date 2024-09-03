import os
import yaml
import torch
import wandb
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from chat_summarization.dataset import Preprocess, prepare_test_dataset

def load_tokenizer_and_model_for_test(config, device):
    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)  # 모델을 GPU로 이동

    return generate_model, tokenizer

def inference(config):
    # Initialize wandb for tracking the inference process
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=f"{config['wandb']['name']}_inference",
        config=config
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device)

    # Prepare test dataset
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []

    def inference_step(engine, batch):
        with torch.no_grad():
            text_ids.extend(batch['ID'])
            generated_ids = generate_model.generate(
                input_ids=batch['input_ids'].to(device),  # 배치를 GPU로 이동
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
            )
            decoded_summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            summary.extend(decoded_summaries)
            return decoded_summaries

    # Ignite inference engine
    inference_engine = Engine(inference_step)
    pbar = ProgressBar()
    pbar.attach(inference_engine)

    # Run inference
    inference_engine.run(dataloader, max_epochs=1)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary": preprocessed_summary,
        }
    )

    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    wandb.finish()

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for PLM summarization.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    output = inference(config)
