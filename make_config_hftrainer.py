import yaml
import argparse

def create_config(config_path):
    config = {
        'model_fn': './models/review.native.kcbert.pth',
        'train_fn': './data/review.sorted.uniq.refined.shuf.train.tsv',
        'pretrained_model_name': 'beomi/kcbert-base',
        'use_albert': False,
        'gpu_id': -1,
        'verbose': 2,
        'batch_size': 80,
        'n_epochs': 1,
        'lr': 5e-5,
        'warmup_ratio': 0.2,
        'adam_epsilon': 1e-8,
        'use_radam': False,
        'valid_ratio': 0.2,
        'max_length': 100,
        'batch_size_per_device': 32,   # 추가된 설정 항목
        'top_k': 1,
        'drop_rnn': True,
        'drop_cnn': True,
        'wandb': {
            'entity': 'oompulab',        # Wandb 엔터티 (사용자 또는 팀 이름)
            'project': 'LP-PLM-baseline',     # Wandb 프로젝트 이름
            'name': 'review.native.kcbert.run.01',            # 특정 실행(run)의 이름
            'log_model': False,             # 모델 체크포인트를 Wandb에 기록할지 여부
            'save_code': False,             # 실험 중 코드 스냅샷을 Wandb에 저장할지 여부
            'notes': 'Native kcbert Experiment notes',   # 실험에 대한 간단한 메모
            'tags': ['native.kcber', 'adam'],      # 실행(run)을 구분하기 위한 태그
        }
    }

    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

    print(f"Configuration file saved as {config_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument('--config_path', required=True, help="Path to save the configuration file")

    args = p.parse_args()

    create_config(args.config_path)
