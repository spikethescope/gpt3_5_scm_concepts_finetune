!pip -q install datasets tiktoken openai

# Fine Tuning OpenAI GPT-3.5-turbo

A lot taken from:
https://github.com/openai/openai-cookbook

!pip install openai
import openai
import os

from google.colab import drive
drive.mount('/content/drive')
with open("/content/drive/My Drive/openaiapikey.txt") as file:
    openai.api_key = file.readline()

## Prepare your data

!pip install tiktoken

import json
import os
import tiktoken
import numpy as np
from collections import defaultdict

# I am picking one file here but you would probably want to do a lot more for a proper model
data_path = "/content/answer.jsonl"

####THE FOLLOWING CODE WAS GENERATED USING CODE LLAMA FROM PERPLEXITY LABS
#####VERY GOOD FROM 7B MODEL THAT SOLVED MY PROBLEM WHICH GPT AND BARD COULD NOT.
import json
def find_invalid_chars(json_str):
    try:
        json.JSONDecoder().decode(json_str)
    except json.JSONDecodeError as e:
        print(f"Invalid characters found: {e}")

def clean_data(data):
    """
    Cleans the data to suit it for JSONL conversion.

    Args:
        data (str): The raw data to clean.

    Returns:
        str: The cleaned data in JSONL format.
    """
    # Split the data into lines
    lines = data.split("\n")

    # Initialize an empty list to store the cleaned data
    json_dataset = []

    # Iterate over each line in the data
    for line in lines:
        # Remove any leading or trailing whitespace
        line = line.strip()
        # Remove invalid characters
        find_invalid_chars(line)
        # Find invalid characters
        #invalid_chars = []
        #for char in line:
            #if char not in json.loads(line):
                #invalid_chars.append(char)

        # Remove invalid characters
        #line = line.replace(invalid_chars, '')
        # Check if the line contains a valid JSON object
        try:
            json_object = json.loads(line)

            # Add the JSON object to the list
            json_dataset.append(json_object)

        except json.JSONDecodeError:
            # If the line does not contain a valid JSON object, skip it
            print("line ignored")
            pass

    # Convert the list of JSON objects to a JSONL file
    with open("output.jsonl", "w") as f:
        for json_object in json_dataset:
            f.write(json.dumps(json_object) + "\n")

    return json_dataset

# Read in the raw data from a file
with open("answer_textformat.txt", "r") as f:
    data = f.read()

# Clean the data and convert it to JSONL format
json_dataset = clean_data(data)

# Print the cleaned data
print(json_dataset)

def convert_to_jsonl(json_dataset):
    """
    Converts a list of JSON objects to a JSONL dataset.

    Args:
        json_dataset (list): A list of JSON objects to convert to JSONL format.

    Returns:
        str: The JSONL dataset as a string.
    """
    # Convert the list of JSON objects to a JSONL file
    with open("output.jsonl", "w") as f:
        for json_object in json_dataset:
            f.write(json.dumps(json_object) + "\n")

    # Read the JSONL file and return the contents as a string
    with open("output.jsonl", "r") as f:
        return f.read()

# Convert the JSON dataset to a JSONL dataset
jsonl_dataset = convert_to_jsonl(json_dataset)

# Print the JSONL dataset
print(jsonl_dataset)

### converting the conversation to correct format

# Initial dataset stats
print("Num examples:", len(json_dataset))
print("First example:")
for message in json_dataset[0]["messages"]:
    print(message)

# Format error checks
format_errors = defaultdict(int)

for ex in json_dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant"):
            format_errors["unrecognized_role"] += 1

        content = message.get("content", None)
        if not content or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")

# Token counting functions
encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# Warnings and tokens counts
n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in json_dataset:
    messages = ex["messages"]
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

print("Num examples missing system message:", n_missing_system)
print("Num examples missing user message:", n_missing_user)
print_distribution(n_messages, "num_messages_per_example")
print_distribution(convo_lens, "num_total_tokens_per_example")
print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
n_too_long = sum(l > 4096 for l in convo_lens)
print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

json_dataset[:2]

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(json_dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
print("See pricing page to estimate total costs")


import json

def save_to_jsonl(conversations, file_path):
    with open(file_path, 'w') as file:
        for conversation in conversations:
            json_line = json.dumps(conversation)
            file.write(json_line + '\n')

# train dataset
save_to_jsonl(json_dataset, '/content/scm_task_train.jsonl')

# train dataset
save_to_jsonl(json_dataset[10:13], '/content/scm_task_train_validation.jsonl')

## Upload your data

# curl -https://api.openai.com/v1/files \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -F "purpose=fine-tune" \
#   -F "file=@path_to_your_file"

training_file_name = '/content/scm_task_train.jsonl'
validation_file_name = '/content/scm_task_train_validation.jsonl'

training_response = openai.File.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response["id"]

validation_response = openai.File.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune"
)
validation_file_id = validation_response["id"]

print("Training file id:", training_file_id)
print("Validation file id:", validation_file_id)

## Create a Fine Tuning Job

# curl https://api.openai.com/v1/fine_tuning/jobs \
# -H "Content-Type: application/json" \
# -H "Authorization: Bearer $OPENAI_API_KEY" \
# -d '{
#   "training_file": "TRAINING_FILE_ID",
#   "model": "gpt-3.5-turbo-0613",
# }'

suffix_name = "scm-finetune-test"


response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix=suffix_name,
)

job_id = response["id"]

print(response)

response = openai.FineTuningJob.retrieve(job_id)
print(response)

response = openai.FineTuningJob.list_events(id=job_id, limit=50)

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])


response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]

print(response)
print("\nFine-tuned model id:", fine_tuned_model_id)

## Generating using the new model


test_messages = []
test_messages.append({"role": "system", "content": "You are an helpful expert in the supply chain area"})
user_message = """i will give you a passage. you have to select the concepts used in the passage. Here is the passage: What these technologies didn’t yet provide, though, were transformative capabilities for
supply-chain management: linking and combining
cross-functional data (for example, inventory,
shipments, and schedules) from internal and
external sources; uncovering the origins of
performance problems by delving into ERP,
warehouse-management, advance-planning, and
other systems all at once; or forecasting demand
and performance with advanced analytics, so
planning can become more precise and problems
can be anticipated and prevented."""
test_messages.append({"role": "user", "content": user_message})

print(test_messages)


test_messages = []
test_messages.append({"role": "system", "content": "You are an helpful expert in the supply chain area"})
user_message = """i will give you a passage. you have to select the concepts used in the passage.The production of cars in Europe rely on a “built-to-order” system, where cars with specific
options need to be assembled in relatively short periods of time. The cars to be assembled
are typically diverse in respect to their required options, and hence, the assembly process
is necessarily slow. Since 1993, the car manufacturer, Renault, has been tackling a variant
of “car sequencing” with the aim of assembling a range of diverse cars as efficiently as
possible (Solnon et al., 2008). In 2005, the subject of the ROADEF challenge was the same
car sequencing problem (Nguyen & Cung, 2005), which is one of finding an optimal order,
or sequence, in which cars are to be processed on a production line. Each car has a set of
options such as paint colour, air conditioning, etc. Each option has an associated constraint
that gives the maximum number of cars that can have this option applied in any contiguous
subsequence. The car sequencing problem is known to be NP-hard (Kis, 2004)."""
test_messages.append({"role": "user", "content": user_message})

print(test_messages)

response = openai.ChatCompletion.create(
    model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=150
)
print(response["choices"][0]["message"]["content"])

response

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo', messages=test_messages, temperature=0, max_tokens=500
)
print(response["choices"][0]["message"]["content"])

!pip install nbconvert
#!magic enable --shell
!jupyter nbconvert "/content/drive/MyDrive/Colab Notebooks/SCM_Fine_tuning_OpenAI_GPT_3_5_turbo.ipynb" --to html
#!jupyter nbconvert --to html /content/drive/My%%20Drive/Colab%%20Notebooks/SCM_Fine_tuning_OpenAI_GPT_3_5_turbo.ipynb