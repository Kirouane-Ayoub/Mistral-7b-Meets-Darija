# Mistral 7b Meets Darija 🇩🇿: A Continual Pre-training Journey 

Welcome to the **[Mistral 7b Meets Darija 🇩🇿](https://medium.com/@ayoubkirouane3/mistral-7b-meets-darija-a-continual-pre-training-journey-90e037c5cbef)** project! 

This project aims to develop a **large language model** capable of **understanding** and **generating text** in **Algerian Darija**. We're using **continual pre-training** to teach the model **Arabic** and **French** first, and then **fine-tuning** it on a specifically curated **Algerian Darija dataset**.

#### Technical Approach:
- **Model**: We're primarily using **Mistral-7B**, a powerful and efficient large language model.
- **Training Method**: Continual pre-training on Arabic and French datasets followed by fine-tuning on Algerian Darija data.
- **Tools and Techniques:** Leveraging **PEFT**, **[Unsloth Framework](https://github.com/unslothai/unsloth)**, **RoPE**, and **Flash Attention-2** for optimized training and performance.

The project Details can be found in the This [blog post](https://medium.com/@ayoubkirouane3/mistral-7b-meets-darija-a-continual-pre-training-journey-90e037c5cbef) .


## Running the train Script :

before runing the train script, [You should install the requirements](https://github.com/unslothai/unsloth):

- **For Pytorch 2.1.0** : 
```
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu121

pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"

```
- **For Pytorch 2.1.1:**
```
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton \
  --index-url https://download.pytorch.org/whl/cu121

pip install "unsloth[cu118-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"

```
- **For Pytorch 2.2.0 :** 

```
pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu118-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git" 
```


**RUN** the `train.py` script from your terminal:
```
python src/train.py
```
**NOTE :**  You can change the model and train settings from `settings.py` file.

## Running the Inference Script : 

Run the script from your terminal:
   ```bash
   python inference.py --prompt "وحد نهار" --max_new_tokens 100 
   ```
   - Replace `وحد نهار` with your desired prompt.
   - Adjust the `max_new_tokens` value as needed.

This script will load your pre-trained model, generate text based on your input prompt, and print the results to the console.
