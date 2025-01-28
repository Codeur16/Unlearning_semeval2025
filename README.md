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
pip install git+https://github.com/huggingface/transformers datasets accelerate deepspeed evaluate matplotlib hydra-core trl omegaconf peft rouge_score tqdm einops packaging bitsandbytes scipy ninja
```
### Lancement 
```bash
pip install -r requirements.txt
python3 Unlearn_function.py
```