# Databricks notebook source
# MAGIC %md
# MAGIC 1. Prepare a novel dataset
# MAGIC 1. Fine-tune the T5-small model to classify movie reviews.
# MAGIC 1. Leverage DeepSpeed to enhance training process.

# COMMAND ----------

assert "gpu" in spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"), "THIS LAB REQUIRES THAT A GPU MACHINE AND RUNTIME IS UTILIZED."

# COMMAND ----------

# MAGIC %pip install rouge_score==0.1.2

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Working Directory: {DA.paths.working_dir}")

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a local temporary directory on the Driver. This will serve as a root directory for the intermediate model checkpoints created during the training process. The final model will be persisted to DBFS.

# COMMAND ----------

import tempfile

tmpdir = tempfile.TemporaryDirectory()
local_training_root = tmpdir.name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tuning

# COMMAND ----------

import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

import evaluate
import nltk
from nltk.tokenize import sent_tokenize

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 1: Data Preparation
# MAGIC For the instruction-following use cases we need a dataset that consists of prompt/response pairs along with any contextual information that can be used as input when training the model. The [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) is one such dataset that provides high-quality, human-generated prompt/response pairs. 
# MAGIC
# MAGIC Let's start by loading this dataset using the `load_dataset` functionality.

# COMMAND ----------

# TODO
ds = <FILL_IN>

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_1(ds)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 2: Select pre-trained model
# MAGIC
# MAGIC The model that we are going to fine-tune is [pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped). This model is one of a Pythia Suite of models that have been developed to support interpretability research.
# MAGIC
# MAGIC Let's define the pre-trained model checkpoint.

# COMMAND ----------

# TODO
model_checkpoint = <FILL_IN>

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_2(model_checkpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 3: Load and Configure
# MAGIC
# MAGIC The next task is to load and configure the tokenizer for this model. The instruction-following process builds a body of text that contains the instruction, context input, and response values from the dataset. The body of text also includes some special tokens to identify the sections of the text. These tokens are generally configurable, and need to be added to the tokenizer.
# MAGIC
# MAGIC Let's go ahead and load the tokenizer for the pre-trained model. 

# COMMAND ----------

# TODO
# load the tokenizer that was used for the model
tokenizer = <FILL_IN>
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["### End", "### Instruction:", "### Response:\n"]}
)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_3(tokenizer)

# COMMAND ----------

# MAGIC %md ### Question 4: Tokenize

# COMMAND ----------

# MAGIC %md
# MAGIC The `tokenize` method below builds the body of text for each prompt/response.

# COMMAND ----------

remove_columns = ["instruction", "response", "context", "category"]


def tokenize(x: dict, max_length: int = 1024) -> dict:
    """
    For a dictionary example of instruction, response, and context a dictionary of input_id and attention mask is returned
    """
    instr = x["instruction"]
    resp = x["response"]
    context = x["context"]

    instr_part = f"### Instruction:\n{instr}"
    context_part = ""
    if context:
        context_part = f"\nInput:\n{context}\n"
    resp_part = f"### Response:\n{resp}"

    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{instr_part}
{context_part}
{resp_part}

### End
"""
    return tokenizer(text, max_length=max_length, truncation=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's `tokenize` the Dolly training dataset. 

# COMMAND ----------

# TODO
tokenized_dataset = <FILL_IN>

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_4(tokenized_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 5: Setup Training
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC To setup the fine-tuning process we need to define the `TrainingArguments`.
# MAGIC
# MAGIC Let's configure the training to have **10** training epochs (`num_train_epochs`) with a per device batch size of **8**. The optimizer (`optim`) to be used should be `adamw_torch`. Finally, the reporting (`report_to`) list should be set to *tensorboard*.

# COMMAND ----------

# TODO
checkpoint_name = "test-trainer-lab"
local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)
training_args = <FILL_IN>

# COMMAND ----------

checkpoint_name = "test-trainer-lab"

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_5(training_args)

# COMMAND ----------

# MAGIC %md ### Question 6: AutoModelForCausalLM
# MAGIC
# MAGIC The pre-trained `pythia-70m-deduped` model can be loaded using the [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) class.

# COMMAND ----------

# TODO
# load the pre-trained model
model = <FILL_IN>

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_6(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 7: Initialize the Trainer
# MAGIC
# MAGIC Unlike the IMDB dataset used in the earlier Notebook, the Dolly dataset only contains a single *train* dataset. Let's go ahead and create a [`train_test_split`](https://huggingface.co/docs/datasets/v2.12.0/en/package_reference/main_classes#datasets.Dataset.train_test_split) of the train dataset.
# MAGIC
# MAGIC Also, let's initialize the [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) with model, training arguments, the train & test datasets, tokenizer, and data collator. Here we will use the [`DataCollatorForLanguageModeling`](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling).

# COMMAND ----------

# TODO
# used to assist the trainer in batching the data
TRAINING_SIZE=6000
SEED=42
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)
split_dataset = <FILL_IN>
trainer = <FILL_IN>

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_7(trainer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 8: Train

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Before starting the training process, let's turn on Tensorboard. This will allow us to monitor the training process as checkpoint logs are created.

# COMMAND ----------

tensorboard_display_dir = f"{local_checkpoint_path}/runs"

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

# MAGIC %md
# MAGIC Start the fine-tuning process!

# COMMAND ----------

# TODO
# invoke training - note this will take approx. 30min
<FILL_IN>

# save model to the local checkpoint
trainer.save_model()
trainer.save_state()

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_8(trainer)

# COMMAND ----------

# persist the fine-tuned model to DBFS
final_model_path = f"{DA.paths.working_dir}/llm04_fine_tuning/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)

# COMMAND ----------

import gc
import torch

gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

fine_tuned_model = AutoModelForCausalLM.from_pretrained(final_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Recall that the model was trained using a body of text that contained an instruction and its response. A similar body of text, or prompt, needs to be provided when testing the model. The prompt that is provided only contains an instruction though. The model will `generate` the response accordingly.

# COMMAND ----------

def to_prompt(instr: str, max_length: int = 1024) -> dict:
    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Response:
"""
    return tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)


def to_response(prediction):
    decoded = tokenizer.decode(prediction)
    # extract the Response from the decoded sequence
    m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", decoded, flags=re.DOTALL)
    res = "Failed to find response"
    if m:
        res = m.group(1).strip()
    else:
        m = re.search(r"#+\s*Response:\s*(.+)", decoded, flags=re.DOTALL)
        if m:
            res = m.group(1).strip()
    return res

# COMMAND ----------

import re
# NOTE: this cell can take up to 5mins
res = []
for i in range(100):
    instr = ds["train"][i]["instruction"]
    resp = ds["train"][i]["response"]
    inputs = to_prompt(instr)
    pred = fine_tuned_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=128,
    )
    res.append((instr, resp, to_response(pred[0])))

# COMMAND ----------

pdf = pd.DataFrame(res, columns=["instruction", "response", "generated"])
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC **CONGRATULATIONS**
# MAGIC
# MAGIC You have just taken the first step toward fine-tuning your own slimmed down version of [Dolly](https://github.com/databrickslabs/dolly)! 
# MAGIC
# MAGIC Unfortunately, it does not seem to be too generative at the moment. Perhaps, with some additional training and data the model could be more capable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 9: Evaluation
# MAGIC
# MAGIC Although the current model is under-trained, it is worth evaluating the responses to get a general sense of how far off the model is at this point.
# MAGIC
# MAGIC Let's compute the ROGUE metrics between the reference response and the generated responses.

# COMMAND ----------

nltk.download("punkt")

rouge_score = evaluate.load("rouge")


def compute_rouge_score(generated, reference):
    """
    Compute ROUGE scores on a batch of articles.

    This is a convenience function wrapping Hugging Face `rouge_score`,
    which expects sentences to be separated by newlines.

    :param generated: Summaries (list of strings) produced by the model
    :param reference: Ground-truth summaries (list of strings) for comparison
    """
    generated_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in generated]
    reference_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in reference]
    return rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True,
    )

# COMMAND ----------

# TODO
rouge_scores = <FILL_IN>
display(<FILL_IN>)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_9(rouge_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC Clean Up

# COMMAND ----------

tmpdir.cleanup()
