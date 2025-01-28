# Unlearning_semeval2025

# Projet Unlearning avec LoRA

Ce projet utilise des techniques d'**unlearning** (désapprentissage) pour ajuster un modèle préexistant à l'aide de LoRA (Low-Rank Adaptation). Le modèle est affiné pour oublier certaines informations tout en conservant les connaissances importantes. Ce projet utilise **Hugging Face Transformers**, **PyTorch**, et **PEFT** (Parameter Efficient Fine-Tuning) pour effectuer le désapprentissage.

## Prérequis

Avant de commencer, assurez-vous que votre environnement dispose des prérequis suivants :

- Python 3.x
- PyTorch
- Hugging Face Transformers
- Autres bibliothèques Python nécessaires (voir ci-dessous)

### Installation des dépendances

Pour installer toutes les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```
### Parameters
```bash
Set the model_token:
Unlearn(
    model_path="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
    unlearn_model_save_path="model/Unlearn_model",
    forget_train_path="Data/forget_train.jsonl",
    retain_train_path="Data/retain_train.jsonl",
    model_token = ""
)
```
### Lancement 
```bash
python3 Unlearn_function.py
```
