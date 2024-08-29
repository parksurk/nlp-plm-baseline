import torch
import torch.nn.utils as torch_utils
from ignite.engine import Events
from simple_ntc.utils import get_grad_norm, get_parameter_norm
from simple_ntc.trainer import Trainer, MyEngine

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class EngineForBert(MyEngine):

    def __init__(self, func, model, crit, optimizer, scheduler, config, wandb=None):
        self.scheduler = scheduler
        self.wandb = wandb  # Wandb 객체를 받아 저장
        super().__init__(func, model, crit, optimizer, config, wandb)

    @staticmethod
    def train(engine, mini_batch):
        # Reset gradients
        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = mini_batch['input_ids'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device)
        mask = mini_batch['attention_mask'].to(engine.device)

        x = x[:, :engine.config['max_length']]

        # Forward pass
        y_hat = engine.model(x, attention_mask=mask).logits

        loss = engine.crit(y_hat, y)
        loss.backward()

        # Calculate accuracy if applicable
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Gradient descent step
        engine.optimizer.step()
        engine.scheduler.step()

        # Wandb에 지표 기록
        if engine.wandb:
            engine.wandb.log({
                'train_loss': float(loss),
                'train_accuracy': float(accuracy),
                '|param|': p_norm,
                '|g_param|': g_norm,
            })

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device)
            mask = mini_batch['attention_mask'].to(engine.device)

            x = x[:, :engine.config['max_length']]

            # Forward pass
            y_hat = engine.model(x, attention_mask=mask).logits

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

            # Wandb에 지표 기록
            if engine.wandb:
                engine.wandb.log({
                    'valid_loss': float(loss),
                    'valid_accuracy': float(accuracy),
                })

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }


class BertTrainer(Trainer):

    def __init__(self, config, wandb=None):
        self.config = config
        self.wandb = wandb  # Wandb 객체를 받아 저장

    def train(
        self,
        model, crit, optimizer, scheduler,
        train_loader, valid_loader,
    ):
        train_engine = EngineForBert(
            EngineForBert.train,
            model, crit, optimizer, scheduler, self.config, self.wandb
        )
        validation_engine = EngineForBert(
            EngineForBert.validate,
            model, crit, optimizer, scheduler, self.config, self.wandb
        )

        EngineForBert.attach(
            train_engine,
            validation_engine,
            verbose=self.config['verbose']
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            run_validation,  # function
            validation_engine, valid_loader,  # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            EngineForBert.check_best,  # function
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config['n_epochs'],
        )

        model.load_state_dict(validation_engine.best_model)

        return model
