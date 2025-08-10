# import some libraries
import os
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift #, LoraConfig 
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
import datasets
import re

# =====================
# Config & Hyperparams
# =====================
DATA_SEED = 42
MAX_LENGTH = 8196
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
MODEL_ID_OR_PATH = 'Qwen/Qwen3-8B'

SYSTEM_PROMPT = 'You are a helpful assistant.'
OUTPUT_DIR = '/mnt/disks/sdb/dayoon_ko/output/nqpara-all'
LORA_RANK = 8
LORA_ALPHA = 32
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
GRAD_ACC_STEPS = 32
NUM_EPOCHS = 2
SPLIT_DATASET_RATIO = 0.02
NUM_PROC = 8
DATASET_NAME = 'dayoon/NQPara-All'
SHUFFLE = True

START_PATTERN = [27, 91, 7265, 10716, 13576, 91, 397]
END_PATTERN = [27, 91, 408, 10716, 13576, 91, 1339]
MASK_VALUE = -100

# =====================
# Utility Functions
# =====================

PROMPT = """**Task Instruction:**

You are a reasoning assistant equipped with web search capabilities to accurately answer the user's questions.

Follow these steps:
1. **Clearly identify** the specific information you need to answer the user's question.
2. **Perform a web search** for the required information by writing your queries as follows:
```
<|begin_search_queries|>
Your search queries here (multiple queries can be placed together seperated by ";\n")
<|end_search_queries|>
```
3. Review the provided search results.
4. If additional information is still required, repeat step 2 with new queries.
5. Once all relevant information has been gathered, use your reasoning abilities to synthesize a clear, concise, and accurate answer.

**Remember:**
* Clearly separate each search query.
* Combine multiple queries into a single search action when they can be run simultaneously.
"""


def prepare_output_dir(output_dir):
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def find_all_sublists(lst, sub):
    """Return all indices of sub in lst"""
    indices = []
    i = 0
    while i <= len(lst) - len(sub):
        if lst[i:i+len(sub)] == sub:
            indices.append(i)
            i += len(sub)
        else:
            i += 1
    return indices


def mask_all_between_patterns(example):
    labels = example['labels']
    start_indices = find_all_sublists(labels, START_PATTERN)
    end_indices = find_all_sublists(labels, END_PATTERN)
    mask_ranges = []
    si = 0
    ei = 0
    while si < len(start_indices) and ei < len(end_indices):
        start = start_indices[si] + len(START_PATTERN)
        # end는 start 이후에 나오는 첫 end_pattern
        while ei < len(end_indices) and end_indices[ei] < start:
            ei += 1
        if ei < len(end_indices):
            end = end_indices[ei]
            if start < end:
                mask_ranges.append((start, end))
            si += 1
            ei += 1
        else:
            break
    for start, end in mask_ranges:
        labels[start:end] = [MASK_VALUE] * (end - start)
    example['labels'] = labels
    return example


def get_search_limit_from_response(item):
    """Extract search limit from the last <|begin_search_queries|><|end_search_queries|> pattern in response"""
    start_pattern = "<|begin_search_queries|>"
    end_pattern = "<|end_search_queries|>"
    response = item["response"]
    
    search_limit = item["num_search_calls"]
    start_idx = response.rfind(start_pattern)
    if start_idx == -1:
        return search_limit
    
    end_idx = response.find(end_pattern, start_idx)
    if end_idx == -1:
        return search_limit
    
    search_content = response[start_idx + len(start_pattern):end_idx]
    queries = [q.strip() for q in search_content.split(';\n') if q.strip()]
    return search_limit + len(queries) 

def get_turn_limit_from_response(item):
    return item["num_search_turns"]

def map_func(x):
    return {"messages": [
        {"role": "user", "content": PROMPT + "\n\nPlease answer the question: " + x["integrated_q"]},
        {"role": "assistant", "content": x["response"]}
    ]}


def get_datasets(dataset_name, tokenizer, split_ratio, data_seed, template, num_proc, shuffle=True):
    dataset = datasets.load_dataset(dataset_name)["train"]
    dataset = dataset.map(partial(map_func))
    if shuffle:
        dataset = dataset.shuffle(seed=data_seed)
    split_idx = int(len(dataset) * (1 - split_ratio))
    train_dataset = dataset.select(range(0, split_idx))
    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    train_dataset = train_dataset.map(mask_all_between_patterns)
    val_dataset = dataset.select(range(split_idx, len(dataset)))
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
    val_dataset = val_dataset.map(mask_all_between_patterns)
    return train_dataset, val_dataset


def get_training_args(output_dir):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        max_grad_norm=1.0,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_checkpointing=True,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['wandb'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=2,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=NUM_EPOCHS,
        metric_for_best_model='loss',
        save_total_limit=2,
        logging_steps=64,
        dataloader_num_workers=1,
        data_seed=DATA_SEED,
    )


def prepare_model_and_template(model_id_or_path, system, max_length):
    model, tokenizer = get_model_tokenizer(model_id_or_path, torch_dtype="bfloat16") #"torch.bfloat16" # float32
    template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
    template.set_mode('train')
    model = Swift.prepare_model(model, {})
    return model, tokenizer, template


def main():
    logger = get_logger()
    seed_everything(DATA_SEED)
    output_dir = prepare_output_dir(OUTPUT_DIR)
    logger.info(f'output_dir: {output_dir}')

    # Model & Template
    model, tokenizer, template = prepare_model_and_template(MODEL_ID_OR_PATH, SYSTEM_PROMPT, MAX_LENGTH)
    logger.info(f'model_info: {model.model_info}')
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    # Dataset
    train_dataset, val_dataset = get_datasets(DATASET_NAME, tokenizer, SPLIT_DATASET_RATIO, DATA_SEED, template, NUM_PROC, SHUFFLE)
    logger.info(f'train_dataset: {len(train_dataset)}')
    logger.info(f'val_dataset: {len(val_dataset)}')

    # Training
    logger.info(f"Model: {model}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Dataset Name: {DATASET_NAME}")
    logger.info(f"# Train Dataset: {len(train_dataset)}")
    logger.info(f"# Val Dataset: {len(val_dataset)}")
    
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    training_args = get_training_args(output_dir)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()
    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    os.system(f"python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{MODEL_ID_OR_PATH}').save_pretrained('{last_model_checkpoint}')\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default=MODEL_ID_OR_PATH)
    parser.add_argument("--output_dir", "-o", type=str, default=OUTPUT_DIR)
    parser.add_argument("--dataset_name", "-d", type=str, default=DATASET_NAME)
    parser.add_argument("--learning_rate", "-lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--grad_acc_steps", "-gc", type=int, default=GRAD_ACC_STEPS)
    parser.add_argument("--num_epochs", "-e", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train_batch_size", "-bs", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--shuffle", type=bool, default=True)
    args = parser.parse_args()

    # 하이퍼파라미터를 전역 변수에 반영
    MODEL_ID_OR_PATH = args.model_name
    OUTPUT_DIR = args.output_dir
    LEARNING_RATE = args.learning_rate
    GRAD_ACC_STEPS = args.grad_acc_steps
    NUM_EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    DATASET_NAME = args.dataset_name
    SHUFFLE = args.shuffle
    main()