# based in part on https://seandearnaley.medium.com/elevating-sentiment-analysis-ad02a316df1d
# unsloth code from https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing#scrollTo=2eSvM9zX_2d3

# for import of json file in chatml format: https://github.com/wirytiox/Unsloth-wiry-training-suit/blob/main/unsloth%20notebook%20modifed%20by%20wirytiox.ipynb

#this is also useful
# https://github.com/unslothai/unsloth/wiki

#https://www.analyticsvidhya.com/blog/2024/04/fine-tuning-google-gemma-with-unsloth/  (open in gmail chrome)

# To run the model after look at these:
# https://github.com/ollama/ollama/blob/main/docs/modelfile.md
#https://towardsdatascience.com/set-up-a-local-llm-on-cpu-with-chat-ui-in-15-minutes-4cdc741408df
# ollama create ollama_model_rw_6-10-24 -f Modelfile
# ollama run ollama_model_rw_6-10-24



from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import torch
import os

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported



# Change the working directory to the directory of the script so it saves there
os.chdir(os.path.dirname(os.path.abspath(__file__)))


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Add the LORA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Add out dataset
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports llama-3, zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    print(convos)
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    #print(texts)
    return { "text" : texts, }
pass

from datasets import Dataset
#make sure that value is all in lowercase, otherwise it will complain about nonetype
dataset = Dataset.from_json("reddit_1500_posts_summary_1-20-25_chatml.json")

dataset = dataset.map(formatting_prompts_func, batched = True,)

#dataset[5]["conversations"]
#print(dataset[5]["text"])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
       # max_steps=69,  ## The maximum steps (0 if the epochs are defined) -  if using steps you can try 60, 20, 10. More steps the greater risk of overfit
        num_train_epochs = 2,  #The number of training epochs(0 if the maximum steps are defined).  the more epochs the more risk of overfitting.
        learning_rate=2e-4,  #some use 2e-4 and other 2e-5  with the -5 for if you are getting overfitting issues
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.1, # the higher the value the  less risk of overfitting. started with 0.01 from tutorial above now testing 0.1 but could go to 0.2 
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()


used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# save the model
model.save_pretrained("lora_model_reddit_psychology_1-20-25") # Local saving. This ONLY saves the LoRA adapters, and not the full model. 

#from git import Repo
#repo_url = "https://github.com/ggerganov/llama.cpp.git"
#clone_directory = "/GSEHD/home/rwatkins/Coders_Demo_Jan_2025/llama.cpp"
#os.makedirs(clone_directory, exist_ok=True)
#print(f"Cloning repository from {repo_url} to {clone_directory}...")
#Repo.clone_from(repo_url, clone_directory)
#print("Repository cloned successfully.")

# Save to q4_k_m GGUF
model.save_pretrained_gguf("psychology_1-20-25", tokenizer, quantization_method = "q4_k_m")







