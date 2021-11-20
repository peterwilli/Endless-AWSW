def get_model(name):
    tokenizer = GPT2Tokenizer.from_pretrained(name, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    model = None
    if name == 'distilgpt2':
        model = GPT2LMHeadModel.from_pretrained(name, pad_token_id = tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    else:
        model = GPTNeoForCausalLM.from_pretrained(name, pad_token_id = tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def random_model_folder():
    now = int(time.time())
    models_dir = os.path.join(work_dir, "models", str(now))
    if not os.path.isdir(models_dir):
        pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    return models_dir

def objective(trial):
    model_name = trial.suggest_categorical('model_name', ['EleutherAI/gpt-neo-125M', 'distilgpt2'])
    model, tokenizer = get_model(model_name)
    named_parameters = list(model.named_parameters())
    dataset = get_dataset(tokenizer)
    lr = trial.suggest_float('lr', 0.0001, 0.01)
    batch_size = 64
    train_len = len(dataset['train'])
    num_training_steps = math.ceil(train_len / batch_size)
    num_epoch = 10
    num_total_steps = num_training_steps * num_epoch
    num_warmup_steps = num_training_steps * trial.suggest_int('warmup_factor', 1, math.ceil(num_epoch * 0.1))
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler_str = trial.suggest_categorical('scheduler', ['cosine_schedule_with_warmup', 'cosine_with_hard_restarts_schedule_with_warmup', 'polynomial_decay_schedule_with_warmup'])
    scheduler = None
    if scheduler_str == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps)
    elif scheduler_str == "cosine_with_hard_restarts_schedule_with_warmup":
        cycles = trial.suggest_int('cycles', 1, 10)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps, cycles)
    elif scheduler_str == "polynomial_decay_schedule_with_warmup":
        lr_end = trial.suggest_float('lr_end', 0, 1e-5)
        power = trial.suggest_float('power', 0.1, 0.6)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps, power=power, lr_end=lr_end)

    last_loss = None
        
    class AWSWTrainer(Trainer):
        def _get_train_sampler(self):
            return None

    class AWSWTrainerCallback(TrainerCallback):
        def __init__(self, optimizer):
            self.old_freeze_part_layers = False
            self.optimizer = optimizer

        def on_train_end(self, args, state, control, **kwargs):
            nonlocal last_loss
            learning_rate_history = [h['learning_rate'] for h in state.log_history if 'learning_rate' in h]
            loss_history = [h['loss'] for h in state.log_history if 'loss' in h]
            fig, axs = plt.subplots(2)
            fig.suptitle('Learning rate and loss')
            axs[0].plot(learning_rate_history)
            axs[1].plot(loss_history)
            last_loss = loss_history[-1]

        def on_step_begin(self, args, state, control, **kwargs):
            # Freeze a part
            learning_rate = self.optimizer.param_groups[0]['lr']
            freeze_layer_rate = trial.suggest_float('freeze_layer_rate', 0.0005, 0.001)
            freeze_part_layers = learning_rate > freeze_layer_rate
            if self.old_freeze_part_layers is not freeze_part_layers:
                print(f"set freeze_part_layers: {freeze_part_layers}")
                to_freeze_count = trial.suggest_int('to_freeze_count', 10, 50)
                for name, param in named_parameters[:to_freeze_count * -1]:
                    param.requires_grad = not freeze_part_layers
                self.old_freeze_part_layers = freeze_part_layers

    def train(model, dataset):
        training_args = TrainingArguments(
            random_model_folder(),
            seed=seed,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epoch,
            logging_steps=50,
        )
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=dataset['train'],
            optimizers=(optimizer, scheduler),
            callbacks=[AWSWTrainerCallback(optimizer)]
        )
        trainer.train()
        del training_args
        del trainer
        gc.collect()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()

    train(model, dataset)
    del model
    del dataset
    del tokenizer
    del optimizer
    return last_loss