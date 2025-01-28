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


class Unlearn_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Si "factor" n'est pas dans les entrées, on utilise la méthode de perte par défaut
        if "factor" not in inputs.keys():
            print("Factor error !!!")
            return super().compute_loss(model, inputs, return_outputs)

        factors = inputs.pop("factor")

        # Application des transformations selon la méthode unlearn

        factors = torch.where(factors == -1, torch.tensor(-0.4), factors)
        factors = torch.where(factors == 1, torch.tensor(2.0), factors)

        # Passe avant avec le modèle
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Calcul de la perte
        loss_fct = CrossEntropyLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        valid_counts = (shift_labels != -100).sum(dim=-1).float()

        # Reshape et calcul de la perte ajustée
        loss = loss.view(shift_logits.size(0), -1)
        loss = loss.sum(dim=-1) / valid_counts

        # Ajustement de la perte avec les facteurs
        adjusted_loss = (loss * factors).mean()

        # Retour de la perte ajustée et des sorties si nécessaire
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # Utilisation de SequentialSampler pour l'entraînement
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspection de la signature de la méthode forward du modèle pour garder uniquement les arguments acceptés
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())

            # Gestion des labels, peut être nommée "label" ou "label_ids"
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append('factor')
class Unlearn_TrainerDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "factor" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float32, requires_grad=True, device=batch['input_ids'].device)
        return batch



""" ============================================= dessaprentissage function ========================================="""




def Unlearn(model_path, unlearn_model_save_path, forget_train_path, retain_train_path, model_token):
    """
    =================================== Load model and tokenizer =============================================
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
        per_device_train_batch_size=1,
        seed=100,
        gradient_checkpointing=True,
        merge_weights=True,
        gradient_accumulation_steps = 16,
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
        logging_steps=5,
        save_strategy="steps",
        learning_rate=args.lr,
        fp16=True,
        max_steps=10,
        seed=args.seed,
        logging_dir=f"{unlearn_model_save_path}/logs"
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
    =================================== Data preprocessing =============================================
    """
    def LoadData(data_path):
        if not isinstance(data_path, str):
            raise TypeError(f"data_path doit être une chaîne de caractères, mais {type(data_path).__name__} a été reçu.")

        file_extension = os.path.splitext(data_path)[1].lower()

        if file_extension in ['.json', '.jsonl']:
            # Pour les fichiers JSON ou JSONL
            dataset = load_dataset("json", data_files=data_path, split='train')
        elif file_extension == '.parquet':
            # Pour les fichiers Parquet
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
    model_max_length = 500
    batch_size = 2
    F_dataset = tokenize_dataset_with_factor(forget_train_path, tokenizer, model_max_length)
    R_dataset = tokenize_dataset_with_factor(retain_train_path, tokenizer, model_max_length)
    R_F_dataset_tokenizer = tokenize_dataset_with_factor(R_F_dataset, tokenizer, model_max_length)
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

    forget_loader = DataLoader(F_Tensor_dataset, batch_size=batch_size)
    retain_loader = DataLoader(R_Tensor_dataset, batch_size=batch_size)


    """
    =================================== Calculation of LoRA gradients  =============================================
    """
    def compute_gradients(model, dataloader):
        gradient_info = {name: torch.zeros_like(param) for name, param in model.named_parameters() if "lora" in name}
        # print(f"Gradient_info: {gradient_info}")
        for batch in dataloader:
            input_ids, attention_mask, labels, factor = batch
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if name in gradient_info:
                    gradient_info[name] += param.grad.detach().clone()
            model.zero_grad
            torch.cuda.empty_cache()  # Libérer la mémoire GPU inutilisée

        for name in gradient_info:
            gradient_info[name] /= len(dataloader)
        return gradient_info


    
    # Calcul des gradients pour la rétention et l'oubli
    print("Calcul des gradients pour les données de rétention...")
    retain_gradients = compute_gradients(model, retain_loader)

    print("Calcul des gradients pour les données d'oubli...")
    forget_gradients = compute_gradients(model, forget_loader)


    """
    =================================== calcule la similarité cosinus entre les gradients des deux jeux de données (rétention et oubli) quel point les gradients calculés à partir des deux ensembles de données sont similaires. =============================================
    """
    def compute_cosine_similarity(gradients_1, gradients_2):
        similarity_scores = {}
        for name in gradients_1:
            grad_1 = gradients_1[name].flatten().unsqueeze(0).cpu().numpy()
            grad_2 = gradients_2[name].flatten().unsqueeze(0).cpu().numpy()
            similarity_scores[name] = cosine_similarity(grad_1, grad_2)[0][0]
        return similarity_scores
     # Calculer la similarité cosinus
    similarity_scores = compute_cosine_similarity(retain_gradients, forget_gradients)

    """
    =================================== Locate the critical regions =============================================
    """

    # Localiser les régions critiques
    # Define sim_thresh and grad_thresh here
    sim_thresh = 0.99  # Define the similarity threshold
    grad_thresh = 6e-4 # Define the gradient magnitude threshold

    critical_regions = []
    for name, sim_score in similarity_scores.items():
        grad_magnitude = forget_gradients[name].abs().mean().item()
        if sim_score < sim_thresh and grad_magnitude > grad_thresh:
            critical_regions.append(name)
    del forget_gradients
    print(f"Nombre de régions critiques trouvées : {len(critical_regions)}")
    """
    ===================================  Filter the model parameters affected by critical regions  =============================================
    """
    # Filtrer les parametres du models avant la mise a jour des poids
    for n, p in model.named_parameters():
        if n in critical_regions:
            p.requires_grad = True
        else:
            p.requires_grad = False

    """
    ===================================  Modify the model parameters to forget the critical regions.  =============================================
    """
    # Modifier les paramètres du modèle pour oublier les régions critiques
    for name in critical_regions:
        param = dict(model.named_parameters())[name]
        param.data.zero_()  # Effacer les gradients pour ces régions

    """
    =================================== Definition of the optimizer and the scheduler =============================================
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=150,
    )

    """===================================== ulearn trainer ==================================================="""
    Unlearn_trainer_instance = Unlearn_Trainer(
        model=model,
        args=args_training,
        train_dataset=Dataset.from_dict(R_F_dataset_tokenizer ),
        # train_dataset=Dataset.from_dict(F_dataset),R_F_dataset_tokenizer 
        tokenizer=tokenizer,
        data_collator = Unlearn_TrainerDataCollator(tokenizer),
        optimizers=(optimizer, scheduler),
    )

     # Définir le GradScaler
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler(device='cuda')
    
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

        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss.item()}")

    end_time = time.time()
    running_time = end_time - start_time
    hours, remainder = divmod(running_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"End ... : End time: [{start_time}]\n Running time: {hours}:{minutes}:{seconds}")
    """===================================== Save the unlearn model  ==================================================="""

    def create_path_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Répertoire créé : {path}")
        else:
            print(f"Le répertoire existe déjà : {path}")


    unlearn_model_save_path = "/kaggle/working"
    save_path =f"{unlearn_model_save_path}/model" 
    save_path_trainer =f"{unlearn_model_save_path}/model-trainer" 
    # check the output path
    create_path_if_not_exists(save_path)
    create_path_if_not_exists(save_path_trainer)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    Unlearn_trainer_instance.save_model(save_path_trainer)
    tokenizer.save_pretrained(save_path_trainer)
    print("Le modèle a été sauvegardé avec succès.")
    return model, Unlearn_trainer_instance


# Call the unlearn  function
model, Unlearn_trainer_instance = Unlearn(
    model_path="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
    unlearn_model_save_path="/kaggle/working/",
    forget_train_path="/kaggle/input/unlearning/forget_train.jsonl",
    retain_train_path="/kaggle/input/unlearning/retain_train.jsonl",
    model_token = "model_token"
)
