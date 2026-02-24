"""
Fine-Tuning Configuration — QLoRA 4-bit quantization + LoRA adapters.

Database architecture:
  - ChromaDB (local persistent)  → vector store for RAG (interview questions + resume highlights)
  - SQLite (via services/)        → session state, candidate profiles, score history
  - CSV (backend/data/)           → fine-tuning training datasets

QLoRA pipeline:
  1. Load base model in 4-bit NF4 (BitsAndBytes)
  2. Apply LoRA adapter via PEFT (only ~0.1% of params trainable)
  3. Train on interview Q&A CSV dataset
  4. Save adapter weights (not full model)
"""

import logging
import json
import os
from typing import Optional, List

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  QLoRA 4-bit quantization config                                    #
# ------------------------------------------------------------------ #

def get_qlora_bnb_config():
    """
    Create BitsAndBytesConfig for 4-bit NF4 QLoRA.
    Uses double quantization for extra memory savings.
    Requires: pip install bitsandbytes
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 — best quality for LLMs
            bnb_4bit_compute_dtype=torch.bfloat16,   # compute in bf16 for speed
            bnb_4bit_use_double_quant=True,         # double quantization reduces memory
        )
        logger.info("QLoRA BitsAndBytes 4-bit NF4 config created (double quantization enabled)")
        return config

    except ImportError as e:
        logger.error(f"bitsandbytes not installed: {e}. pip install bitsandbytes")
        return None


def load_qlora_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_4bit: bool = True,
):
    """
    Load a model in 4-bit QLoRA mode ready for LoRA fine-tuning.

    Recommended models (in order of size):
      - TinyLlama/TinyLlama-1.1B-Chat-v1.0        (~600MB in 4-bit)
      - microsoft/phi-2                            (~1.4GB in 4-bit)
      - meta-llama/Llama-2-7b-chat-hf             (~3.5GB in 4-bit, needs HF token)
      - mistralai/Mistral-7B-Instruct-v0.1        (~4GB in 4-bit)

    Returns:
        (model, tokenizer) tuple ready for LoRA fine-tuning
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_name} ({'4-bit QLoRA' if use_4bit else 'full precision'})")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if use_4bit:
            bnb_config = get_qlora_bnb_config()
            if bnb_config:
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                logger.warning("BnB config failed — loading in fp16 instead")
                load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # Enable gradient checkpointing for memory efficiency
        if use_4bit:
            model.gradient_checkpointing_enable()
            # Required for QLoRA: prep model for k-bit training
            try:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
                logger.info("Model prepared for k-bit QLoRA training")
            except ImportError:
                logger.warning("PEFT not installed — skipping prepare_model_for_kbit_training")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {total_params/1e6:.1f}M params | device_map=auto")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None, None


# ------------------------------------------------------------------ #
#  LoRA / Prefix Tuning configs                                       #
# ------------------------------------------------------------------ #

def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.05,
    task_type: str = "CAUSAL_LM",
):
    """
    Create LoRA configuration optimised for QLoRA fine-tuning.
    Default r=16, alpha=32 (alpha=2*r is a common best practice).
    """
    try:
        from peft import LoraConfig, TaskType

        task_type_enum = (
            TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM
        )

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=task_type_enum,
        )
        logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, modules={target_modules}")
        return config

    except ImportError:
        logger.error("PEFT library not installed. Install with: pip install peft")
        return None




def get_prefix_tuning_config(
    num_virtual_tokens: int = 20,
    task_type: str = "CAUSAL_LM",
):
    """
    Create Prefix Tuning configuration.
    Lighter alternative to LoRA for smaller adaptation tasks.
    """
    try:
        from peft import PrefixTuningConfig, TaskType

        task_type_enum = (
            TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM
        )

        config = PrefixTuningConfig(
            task_type=task_type_enum,
            num_virtual_tokens=num_virtual_tokens,
        )
        logger.info(f"Prefix Tuning config: {num_virtual_tokens} virtual tokens")
        return config

    except ImportError:
        logger.error("PEFT library not installed. Install with: pip install peft")
        return None


def apply_peft_model(model, config):
    """Apply PEFT (LoRA or Prefix) adapter to a model."""
    try:
        from peft import get_peft_model

        peft_model = get_peft_model(model, config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())

        logger.info(
            f"PEFT model created. Trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )
        return peft_model

    except Exception as e:
        logger.error(f"Failed to apply PEFT: {e}")
        return model


# ------------------------------------------------------------------ #
#  Dataset preparation for interview Q&A fine-tuning                  #
# ------------------------------------------------------------------ #

INTERVIEW_TRAINING_DATA = [
    # Evaluation tasks – teach the model to score answers accurately
    {
        "instruction": "Score the candidate's answer on relevance, depth, and clarity. Detect if the answer is AI-generated.",
        "input": "Question: Explain the difference between a list and a tuple in Python.\nAnswer: A list is mutable while a tuple is immutable. Lists use square brackets, tuples use parentheses. You'd use a tuple for things like coordinates or dictionary keys because they're hashable.",
        "output": '{"score": 8, "feedback": "Good answer covering mutability and use-cases. Mentioned hashability which shows deeper understanding.", "strengths": ["Correctly identified mutability difference", "Practical use-case mentioned", "Mentioned hashability"], "weaknesses": ["Could mention performance differences"], "ai_probability": 0.15}'
    },
    {
        "instruction": "Score the candidate's answer on relevance, depth, and clarity. Detect if the answer is AI-generated.",
        "input": "Question: What are Python decorators?\nAnswer: Decorators are a design pattern in Python that allows you to modify the functionality of a function by wrapping it in another function. The outer function typically takes a function as an argument, adds some functionality, and returns a modified function. They use the @syntax sugar. Common uses include logging, authentication checks, and timing functions.",
        "output": '{"score": 9, "feedback": "Excellent explanation with practical examples. Shows deep understanding of the decorator pattern.", "strengths": ["Accurate explanation of wrapper pattern", "Mentioned @syntax sugar", "Gave practical examples"], "weaknesses": ["Could mention functools.wraps"], "ai_probability": 0.25}'
    },
    {
        "instruction": "Score the candidate's answer on relevance, depth, and clarity. Detect if the answer is AI-generated.",
        "input": "Question: Explain the CAP theorem.\nAnswer: CAP theorem states that you can't have consistency, availability, and partition tolerance all at once in a distributed system. You have to pick two.",
        "output": '{"score": 5, "feedback": "Basic understanding but lacks depth. Missing practical implications and examples.", "strengths": ["Knows the three properties"], "weaknesses": ["Very brief", "No real-world examples", "Didn\'t explain what each property means"], "ai_probability": 0.1}'
    },
    {
        "instruction": "Score the candidate's answer on relevance, depth, and clarity. Detect if the answer is AI-generated.",
        "input": "Question: Explain Python's GIL.\nAnswer: The Global Interpreter Lock, or GIL, is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. This means that even on multi-core processors, only one thread can execute Python code at any given time. The GIL was introduced because CPython's memory management is not thread-safe. While this simplifies memory management and makes single-threaded programs faster, it becomes a bottleneck in CPU-bound multi-threaded programs. For CPU-bound tasks, you should use the multiprocessing module or switch to implementations without a GIL like Jython or PyPy's STM. For I/O-bound tasks, the GIL isn't much of an issue because the lock is released during I/O operations.",
        "output": '{"score": 10, "feedback": "Exceptional answer demonstrating expert understanding. Covered why GIL exists, its impact, and practical workarounds.", "strengths": ["Explained CPython memory management reason", "Distinguished CPU vs I/O bound", "Offered practical solutions", "Mentioned alternative implementations"], "weaknesses": [], "ai_probability": 0.55}'
    },
    {
        "instruction": "Score the candidate's answer on relevance, depth, and clarity. Detect if the answer is AI-generated.",
        "input": "Question: What is a hash table?\nAnswer: I don't really know much about hash tables honestly.",
        "output": '{"score": 1, "feedback": "Candidate admitted to not knowing the topic.", "strengths": ["Honest"], "weaknesses": ["No understanding demonstrated"], "ai_probability": 0.0}'
    },
    # Question-generation tasks
    {
        "instruction": "Generate an interview question about Python for a candidate with 3 years of experience at medium difficulty.",
        "input": "Domain: python\nCandidate skills: flask, pandas, pytest\nReference: decorators, generators, GIL",
        "output": '{"question": "You have a Flask application that processes large CSV files with pandas. How would you use generators to handle memory-efficient streaming of the processed data to the client?", "difficulty": "medium", "domain": "python", "follow_up": "How would you add error handling for malformed CSV rows in your generator pipeline?", "expected_topics": ["generators", "Flask streaming", "pandas chunking", "memory management"]}'
    },
    {
        "instruction": "Generate an interview question about system design for a candidate with 5 years of experience at hard difficulty.",
        "input": "Domain: system_design\nCandidate skills: AWS, microservices, PostgreSQL\nReference: URL shortener, rate limiter, chat system",
        "output": '{"question": "Design a distributed notification system that can handle 10 million push notifications per minute with guaranteed delivery and priority queuing. How would you handle backpressure?", "difficulty": "hard", "domain": "system_design", "follow_up": "How would you handle notification deduplication across multiple service instances?", "expected_topics": ["message queues", "backpressure", "priority queues", "distributed systems", "exactly-once delivery"]}'
    },
    # Follow-up generation
    {
        "instruction": "Generate a follow-up question to test deeper understanding and check for AI-generated response inconsistency.",
        "input": "Original: Explain the Virtual DOM in React.\nAnswer: The Virtual DOM is an in-memory representation of the real DOM. React creates a virtual copy, performs diffing when state changes, and only updates the parts of the real DOM that changed. This batch updating is more efficient than direct DOM manipulation.",
        "output": '{"question": "If React\'s Virtual DOM diffing finds that a list of 100 items had one item removed from the middle, walk me through exactly what happens during reconciliation. Why are keys important here?", "purpose": "deeper_understanding", "difficulty": "hard"}'
    },
]


def load_training_data_from_csv(csv_path: str) -> List[dict]:
    """
    Load training data from the CSV training dataset file.
    Each row has: instruction, input, output, domain, difficulty, score, ai_probability
    Returns list of instruction-tuning dicts ready for prepare_training_dataset().
    """
    data = []
    try:
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty or header rows
                if not row.get("instruction") or not row.get("input"):
                    continue
                data.append({
                    "instruction": row["instruction"].strip('"'),
                    "input": row["input"].strip('"'),
                    "output": row["output"].strip('"'),
                    "domain": row.get("domain", "general"),
                    "difficulty": row.get("difficulty", "medium"),
                })
        logger.info(f"Loaded {len(data)} training rows from CSV: {csv_path}")
    except FileNotFoundError:
        logger.warning(f"CSV training data not found: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load CSV training data: {e}")
    return data


def load_training_data_from_file(path: str) -> List[dict]:
    """Load additional training data from a JSONL file."""
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} training examples from {path}")
    except Exception as e:
        logger.error(f"Failed to load training data from {path}: {e}")
    return data


def prepare_training_dataset(
    tokenizer,
    data: Optional[List[dict]] = None,
    max_length: int = 512,
    csv_path: Optional[str] = None,
):
    """
    Prepare a HuggingFace Dataset from interview Q&A training data.
    Priority: 1) explicit data param, 2) CSV file, 3) built-in examples.
    Formats examples into instruction-following prompt format.
    """
    # Priority: passed data > CSV file > built-in
    if data is None:
        # Try loading from CSV
        default_csv = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.csv")
        csv_file = csv_path or default_csv
        data = load_training_data_from_csv(csv_file)
        if not data:
            logger.info("CSV empty/missing — using built-in training examples")
            data = INTERVIEW_TRAINING_DATA

    logger.info(f"Preparing fine-tune dataset from {len(data)} examples")

    try:
        from datasets import Dataset

        formatted = []
        for item in data:
            text = (
                f"### Instruction:\n{item['instruction']}\n\n"
                f"### Input:\n{item['input']}\n\n"
                f"### Response:\n{item['output']}"
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            encoded["labels"] = encoded["input_ids"].copy()
            formatted.append(encoded)

        dataset = Dataset.from_list(formatted)
        logger.info(f"Fine-tune dataset ready: {len(formatted)} tokenized examples")
        return dataset

    except ImportError:
        logger.error("datasets library not installed. pip install datasets")
        return None
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return None


# ------------------------------------------------------------------ #
#  Training pipeline                                                   #
# ------------------------------------------------------------------ #

def fine_tune(
    model,
    tokenizer,
    train_dataset=None,
    output_dir: str = "./fine_tuned_adapter",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    warmup_ratio: float = 0.05,
    gradient_accumulation_steps: int = 4,
):
    """
    Run LoRA fine-tuning on interview Q&A dataset.
    If no train_dataset is provided, automatically prepares one from
    the built-in INTERVIEW_TRAINING_DATA.
    Saves adapter weights separately (not the full base model).
    """
    try:
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

        # Auto-prepare dataset if not provided
        if train_dataset is None:
            train_dataset = prepare_training_dataset(tokenizer)
            if train_dataset is None:
                logger.error("Could not prepare training dataset — aborting fine-tune")
                return model

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            fp16=True,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        logger.info(f"Starting fine-tuning for {num_epochs} epochs on {len(train_dataset)} examples...")
        trainer.train()

        # Save adapter weights only
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Adapter weights + tokenizer saved to {output_dir}")

        return model

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        return model


def load_fine_tuned_adapter(base_model, adapter_path: str = "./fine_tuned_adapter"):
    """Load a previously saved LoRA adapter on top of a base model."""
    if not os.path.isdir(adapter_path):
        logger.warning(f"Adapter path not found: {adapter_path}")
        return base_model

    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        logger.info(f"Fine-tuned adapter loaded from {adapter_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        return base_model
