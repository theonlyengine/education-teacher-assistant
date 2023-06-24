---
license: apache-2.0
tags:
- MerlynMind
- education
---

# Merlyn-education-teacher-assistant

Merlyn-education-teacher-assistant is a 12b parameter decoder-style transformer model for the education domain. It is fine-tuned from a [pythia-12b](https://huggingface.co/EleutherAI/pythia-12b) base-model.

This model was trained by [Merlyn Mind](https://www.merlyn.org/).

Merlyn-education-teacher-assistant is part of the family of Merlyn Mind models designed specifically for use in in- and out-of-classroom education. 

Merlyn-education-teacher-assistant makes helpful recommendations based on the ongoing classroom discussion, suggesting research activities and topics for further exploration.

## Model Date

June 26, 2023

## Model License

Apache-2.0

## Documentation

* [Merlyn Mind’s education-specific language models](https://www.merlyn.org/)

## Usage

Loading model and tokenizer:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "MerlynMind/merlyn-education-teacher-assistant"
device = torch.device("cuda:0") # change device id as necessary
model = AutoModelForCausalLM.from_pretrained(model_path)    
tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
model.to(device) # move to device
```

Prompt example:

```python
conversation = ''''user1':\tHow do some gases help keep the Earth warm?
'user2':\tSome gases, called greenhouse gases, act like a blanket around Earth by trapping heat from the sun in the atmosphere, which keeps our planet warm. This process is known as the greenhouse effect.
'user1':\tHow can we reduce greenhouse gas emissions?
'user2':\tWe can reduce greenhouse gas emissions by using renewable energy sources, increasing energy efficiency, and reducing waste.'''

prompt = tokenizer.bos_token
prompt += '''Instruction:\tYou are teaching high school students.
Instruction:\tYou are observing the following conversation between two users.
Instruction:\tGenerate 3 research activities based on the conversation.
Instruction:\tThe research activities should be doable by high school students.
Instruction:\tYour response should be a well-formed JSON array of 3 objects, each with a 'title' property and an 'activity' property.

Conversation:''' + f"\n{conversation}" + " Response:"
```

Inference:

```python
inputs = tokenizer(prompt, return_tensors="pt").to(device)
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.0,
    num_beams=2
)
response = tokenizer.decode(generate_ids[0],
                      skip_special_tokens=True,
                      clean_up_tokenization_spaces=True)
```

Example output (after response processing):

```json
[
{"title": "Understanding the Greenhouse Effect", "activity": "Research the greenhouse effect and the role of greenhouse gases in keeping Earth warm. Create a presentation or poster explaining the greenhouse effect and how greenhouse gases act as a blanket around Earth."},
{"title": "Renewable Energy Sources", "activity": "Identify different renewable energy sources, such as solar, wind, and geothermal energy, and explain how they can help reduce greenhouse gas emissions."},
{"title": "Energy Efficiency and Waste Reduction", "activity": "Research energy efficiency and waste reduction practices, and develop a plan to implement these practices in your school or community to reduce greenhouse gas emissions."}
]
```

## Citation

To cite this model, please use:

```
@online{MerlynEducationModels,
    author    = {Merlyn Mind AI Team},
    title     = {Merlyn Mind's education-domain language models},
    year      = {2023},
    url       = {merlyn.org},
    urldate   = {2023-06-26}
}
```