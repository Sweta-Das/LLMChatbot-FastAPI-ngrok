import os
import torch
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import LlamaTokenizerFast, pipeline, LlamaForCausalLM, BitsAndBytesConfig
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory

# Specifying the model name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Quantization Configuration for better memory use
quant_config = BitsAndBytesConfig(
    # Loading model in 4 bits
    load_in_4bit=True,
    # Specifying bnb quantization type
    bnb_4bit_quant_type='nf4',
    # Specifying datatype
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Tokenization and Model
tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    low_cpu_mem_usage=True
)

# Creating pipeline
pipeline = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    num_beams = 10,
    early_stopping = True,
    no_repeat_ngram_size = 2
)

llm = HuggingFacePipeline(pipeline = pipeline)

# Creating chatbot with Regular Memory
template = """
<<SYS>>
You are an AI support assistant having conversation with human. Provide best response as per your knowledge.
<</SYS>>

Previous conversation: {chat_history}

Human: {question}
AI:
"""
prompt2 = PromptTemplate.from_template(template)
memory = ConversationBufferMemory(memory_key='chat_history')
conversation = LLMChain(
    llm = llm,
    prompt = prompt2,
    verbose = True,
    memory = memory
)