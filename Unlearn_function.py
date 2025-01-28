# # # Environnement
#!pip  install git+https://github.com/huggingface/transformers datasets accelerate deepspeed evaluate matplotlib hydra-core trl omegaconf peft rouge_score tqdm einops packaging bitsandbytes scipy ninja


"======================================= V4 ======================================"
from collections import namedtuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
)
from datasets import concatenate_datasets, Dataset
from peft import get_peft_model, LoraConfig, TaskType  # Nécessite `peft`
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer, DataCollatorWithPadding
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
import inspect
import json
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
from tqdm import tqdm

# # Environnement
#!pip  install git+https://github.com/huggingface/transformers datasets accelerate deepspeed evaluate matplotlib hydra-core trl omegaconf peft rouge_score tqdm einops packaging bitsandbytes scipy ninja

class Unlearn_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if "factor" not in inputs.keys():
            print("Factor error !!!")
            return super().compute_loss(model, inputs, return_outputs)

        factors = inputs.pop("factor")


        factors = torch.where(factors == -1, torch.tensor(-2), factors)
        factors = torch.where(factors == 1, torch.tensor(3.0), factors)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fct = CrossEntropyLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        valid_counts = (shift_labels != -100).sum(dim=-1).float()

        loss = loss.view(shift_logits.size(0), -1)
        loss = loss.sum(dim=-1) / valid_counts

        adjusted_loss = (loss * factors).mean()

        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())

            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append('factor')
class Unlearn_TrainerDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "factor" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float32, requires_grad=True, device=batch['input_ids'].device)
        return batch



""" ============================================= Fonction de dessaprentissage ========================================="""




def Unlearn(model_path, unlearn_model_save_path, forget_train_path, retain_train_path, model_token):
    """
    =================================== Step 1: Load model and tokenizer =============================================
    """

    # Configuration des arguments du modèle
    ArgsParams = namedtuple('ArgsParams', [
        'model_id', 'tokenizer_id', 'model_token', 'hf_rep', 'lr', 'epochs',
        'bf16', 'per_device_train_batch_size', 'seed',
        'gradient_checkpointing', 'merge_weights','gradient_accumulation_steps','weight_decay'
        ])

    args = ArgsParams(
        model_id=model_path,
        tokenizer_id="allenai/OLMo-1B-hf",
        model_token = model_token,
        hf_rep=True,
        lr=1e-4,
        epochs=10,
        bf16=True,
        per_device_train_batch_size=8,
        seed=100,
        gradient_checkpointing=True,
        merge_weights=True,
        gradient_accumulation_steps = 4,
        weight_decay =1e-4
    )
    # Configuration LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Modèle causal (auto-régressif)
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules="all-linear" # Modules spécifiques à adapter
    )

    # Configuration de quantification avec BitsAndBytes
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bits_compute_dtype=compute_dtype
    )
        # Define parameters in TrainingArguments
    args_training = TrainingArguments(
        output_dir=unlearn_model_save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=1,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.lr,
        fp16=True,
        max_steps=10,
        seed=args.seed,
        logging_dir=f"{unlearn_model_save_path}/logs",
        overwrite_output_dir=True # Ceci écrasera le répertoire existant s'il existe.
    )




    # Charger le modèle avec LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map='auto',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model = get_peft_model(model, lora_config)

    # Activer le gradient_checkpointing
    model.gradient_checkpointing_enable()

    #print(f"Nombre de paramètres du modèle avec LoRA : {model.num_parameters()}")

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    """
    =================================== Step 2: Prétraitement des données =============================================
    """
    def LoadData(data_path):
        if not isinstance(data_path, str):
            raise TypeError(f"data_path doit être une chaîne de caractères, mais {type(data_path).__name__} a été reçu.")

        file_extension = os.path.splitext(data_path)[1].lower()

        if file_extension in ['.json', '.jsonl']:
            # Pour les fichiers JSON ou JSONL
            dataset = load_dataset("json", data_files=data_path, split='train')
            #dataset = load_dataset("json", data_files=data_path, split='train[:1%]')
        elif file_extension == '.parquet':
            # Pour les fichiers Parquet
            #dataset = load_dataset("parquet", data_files=data_path, split='train[:10%]')
            dataset = load_dataset("parquet", data_files=data_path, split='train')
        else:
            raise ValueError(f"Type de fichier non pris en charge : {file_extension}")

        return dataset

    ### Combine datasets:

    def merge_datasets_and_delete_first(dataset1, dataset2):
        dataset1 = LoadData(dataset1)
        dataset2 = LoadData(dataset2)
        combined_dataset = concatenate_datasets([dataset1,  dataset2])

        return combined_dataset
    R_F_dataset = merge_datasets_and_delete_first(forget_train_path, retain_train_path)


    #M2
    def tokenize_dataset_with_factor(dataset_path, tokenizer, max_length):
        if not isinstance(dataset_path, str):
            dataset = dataset_path
        else:
            dataset = LoadData(dataset_path)

        tokenized_data = {"input_ids": [], "attention_mask": [], "labels": [], "factor": []}

        for entry in dataset:
            # Tokenisation de l'entrée
            encoded = tokenizer(
                entry["input"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            output_text = entry["output"]
            if not isinstance(output_text, str):
                if isinstance(output_text, list):
                    output_text = " ".join(output_text)
                else:
                    output_text = str(output_text)
            encoded_labels = tokenizer(
                output_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

            # Ajout des données tokenisées et du facteur
            tokenized_data["input_ids"].append(encoded["input_ids"])
            tokenized_data["labels"].append(encoded_labels["input_ids"])
            tokenized_data["attention_mask"].append(encoded["attention_mask"])
            tokenized_data["factor"].append(-1 if entry.get("split", "") == "forget" else 1)

        return tokenized_data

    # Tokenisation
    model_max_length = 400
    batch_size = 4
    F_dataset = tokenize_dataset_with_factor(forget_train_path, tokenizer, model_max_length)
    R_dataset = tokenize_dataset_with_factor(retain_train_path, tokenizer, model_max_length)
    R_F_dataset_tokenizer = tokenize_dataset_with_factor(R_F_dataset, tokenizer, model_max_length)
    # Convert dictionary values to tensors and create TensorDataset
    # In tokenize_dataset_with_factor function, change the TensorDataset creation lines:
    F_Tensor_dataset = TensorDataset(
        torch.tensor(F_dataset["input_ids"]),
        torch.tensor(F_dataset["attention_mask"]),
        torch.tensor(F_dataset["labels"]),
        torch.tensor(F_dataset["factor"], dtype=torch.float32, requires_grad=True)
    )
    R_Tensor_dataset = TensorDataset(
        torch.tensor(R_dataset["input_ids"]),
        torch.tensor(R_dataset["attention_mask"]),
        torch.tensor(R_dataset["labels"]),
        torch.tensor(R_dataset["factor"], dtype=torch.float32, requires_grad=True)
    )
    R_F_dataset_tokenizer_Tensor = TensorDataset(
        torch.tensor(R_dataset["input_ids"]),
        torch.tensor(R_dataset["attention_mask"]),
        torch.tensor(R_dataset["labels"]),
        torch.tensor(R_dataset["factor"], dtype=torch.float32, requires_grad=True)
    )

    # forget_loader = DataLoader(F_Tensor_dataset, batch_size=batch_size)
    # retain_loader = DataLoader(R_Tensor_dataset, batch_size=batch_size)
    """
    =================================== Step8 : Definition de l'optimiseur et de l'ordonnanceur =============================================
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=150,
    )

    """
    =================================== Step 3: Calcul des gradients LoRA =============================================
    """
    scaler = torch.amp.GradScaler('cuda') 
  
    """=====================================Step9: ulearning  ====================================================="""
    def create_path_if_not_exists(path):
      if not os.path.exists(path):
          os.makedirs(path)
          print(f"Répertoire créé : {path}")
      else:
          print(f"Le répertoire existe déjà : {path}")
    create_path_if_not_exists(os.path.join(unlearn_model_save_path, "logs"))
    save_path =f"{unlearn_model_save_path}/model"
    
    Unlearn_trainer_instance = Unlearn_Trainer(
        model=model, 
        args=args_training, 
        train_dataset=Dataset.from_dict(R_F_dataset_tokenizer ), 
        tokenizer=tokenizer, 
        data_collator = Unlearn_TrainerDataCollator(tokenizer), 
        optimizers=(optimizer, scheduler), 
    )

     # Définir le GradScaler
    scaler = torch.cuda.amp.GradScaler()
    # scaler = torch.amp.GradScaler(device='cuda')

    start_time = time.time()
    print(f"Start ... : strat time: [{start_time}]")

    # Utilisation de tqdm pour la boucle sur les époques
    for epoch in range(args.epochs):
        # Envelopper la boucle des batches avec tqdm pour afficher la progression
        with tqdm(Unlearn_trainer_instance.get_train_dataloader(), desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch") as pbar:
            for batch in pbar:
                batch = {k: v.cuda() for k, v in batch.items()}
                loss = Unlearn_trainer_instance.compute_loss(model, batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                # Mise à jour de la barre de progression
                pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{args.epochs}")

    end_time = time.time()
    running_time = end_time - start_time
    hours, remainder = divmod(running_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"End ... : End time: [{start_time}]\n Running time: {hours}:{minutes}:{seconds}")
    """=====================================Step10: Save the unlearn model  ==================================================="""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Le modèle a été sauvegardé avec succès.")


# Call the unlearn  function
Unlearn(
    model_path="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
    unlearn_model_save_path="model/Unlearn_model",
    forget_train_path="Data/forget_train.jsonl",
    retain_train_path="Data/retain_train.jsonl",
    model_token = ""
)
