#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import re
import os
from liger_kernel.transformers import AutoLigerKernelForCausalLM

#prompt模板
PROMPT_TEMPLATE = """ 
You are an AI assistant specialized in generating academic paper abstracts. 
Based on the provided paper introduction, please create a concise, comprehensive, and professional abstract. 
Research Introduction: {introduction} 

Please generate a well-structured abstract that includes: 
- A thorough introduction to the research background
- A clear statement of the research objective or question
- A detailed description of the primary method or methodology used
- Comprehensive coverage of the most significant research findings or discoveries
- An in-depth analysis of the significance, impact, or contribution of the study
- And preserves key terms and meaning

The abstract MUST be about 300-400 words in length, written in formal academic language with English and in the third person.
Ensure the abstract is highly detailed, substantive, and covers all aspects of the research extensively to meet the 300-400 word target.
Important: Limit each sentence to a maximum of 2-3 citation references. Do not list multiple authors consecutively. Use "et al." to simplify citations with multiple authors.

### ABSTRACT START ###
"""

#摘要提取
def extract_abstract(full_output, prompt):
    #1：用分隔標記提取
    abstract_start_marker = "### ABSTRACT START ###"
    possible_end_markers = [
        "### ABSTRACT END ###",
        "### ABDSTRACT END ###",  
        "### Abstract Ends Here ###",
        "### ABSTRACT END ===",
        "###.abstract END###",
        "### ENDS WITH A QUESTION MARK",
        "### ABSTRACT OBJECTIVE OR QUESTION",
        "Finish the abstract here."
        
    ]

    end_pos = len(full_output)
    
    if abstract_start_marker in full_output:
        start_pos = full_output.find(abstract_start_marker) + len(abstract_start_marker)
        # 檢查所有可能的結束標記，找到第一個出現的位置
        for end_marker in possible_end_markers:
            marker_pos = full_output.find(end_marker)
            if marker_pos != -1 and marker_pos < end_pos:
                end_pos = marker_pos
        
        # 提取開始標記到第一個結束標記之間的內容
        return full_output[start_pos:end_pos].strip()
    
    #2：用標準摘要標記提取
    if "\nabstract：" in full_output:
        abstract = full_output.split("\nabstract：")[-1].strip()
        return abstract
    
    #3：移除完整的提示字
    if prompt in full_output:
        abstract = full_output.replace(prompt, "").strip()
        return abstract
    
    #4：用提示詞的結束部分定位摘要開始位置
    prompt_end_marker = "to simplify citations with multiple authors."
    if prompt_end_marker in full_output:
        end_pos = full_output.find(prompt_end_marker)
        end_sentence_pos = full_output.find(".", end_pos)
        if end_sentence_pos != -1:
            abstract = full_output[end_sentence_pos + 1:].strip()
            return abstract
    
    #5：使用re處理更複雜的情況，嘗試找出提示詞和回應之間的分界
    abstract_patterns = [
        r"This\s+(?:study|research|paper|work)\s+(?:investigates|examines|explores|analyzes)",
        r"In\s+this\s+(?:study|research|paper|work)",
        r"The\s+(?:aim|purpose|objective|goal)\s+of\s+this\s+(?:study|research|paper|work)",
        r"This\s+(?:paper|article|research|study)\s+(?:presents|reports|describes|introduces)"
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, full_output)
        if match:
            abstract = full_output[match.start():].strip()
            return abstract
    
    # 默認return原始輸出
    return full_output.strip()

#修正引用鏈的函數
def fix_citation_chains(text):
    """修正生成摘要中有問題的引用鏈"""
    #匹配大量括號內引用鏈的模式
    pattern = r'\(([^()]*?;[^()]*?;[^()]*?;[^()]*?)\)'
    
    #尋找所有匹配
    matches = re.findall(pattern, text)
    
    #處理每個包含超過2個引用的匹配
    for match in matches:
        if match.count(';') > 1:
            #提取前幾個引用
            citations = match.split(';')[:2]
            replacement = '(' + ';'.join(citations) + ' et al.)'
            #取代冗長的引用鏈
            text = text.replace('(' + match + ')', replacement)
    
    #檢查繼續出現名稱和年份的引用鏈
    author_chain_pattern = r'(\([^()]+\d{4}\);\s*[A-Z][a-z]+\s*(?:&|and)\s*[A-Z][a-z]+\s*\(\d{4}\))'
    text = re.sub(author_chain_pattern, r'\1 et al.)', text)
    
    #修正特別長的作者列表
    long_author_list = r'(?:[A-Z][a-z]+(?: & | and )[A-Z][a-z]+(?:, )?){3,}'
    text = re.sub(long_author_list, "et al.", text)
    
    return text

#確保摘要完整性
def ensure_complete_abstract(abstract):
    """確保摘要不會以不完整的引用或作者列表結尾"""
    #檢查摘要是否以開放的括號結尾
    if '(' in abstract and ')' not in abstract[abstract.rfind('('):]:
        #尋找最後一個完整句子
        last_period = abstract.rfind('.')
        if last_period > 0:
            abstract = abstract[:last_period+1]
    
    #檢查可能指示不完整引用列表的模式
    if re.search(r'(?:et al\.|[A-Z][a-z]+)(?:\s*(?:&|and)\s*[A-Z][a-z]+)*\s*\(\d{4}\)$', abstract):
        #尋找最後一個完整句子
        last_period = abstract.rfind('.')
        if last_period > 0 and last_period < len(abstract) - 20:
            abstract = abstract[:last_period+1]
    
    return abstract

#摘要後處理函數
def postprocess_abstract(abstract, paper_id, original_intro, target_word_count=400):
    # 移除可能剩餘的提示詞片段
    prompt_fragments = [
        "You are an AI assistant",
        "Based on the provided paper introduction",
        "Please generate",
        "The abstract should be",
        "Research Introduction:",
        "abstract:",
        "### Abstract:",
        "Background:"
        "ABSTRACT BACKGROUND:",
        "### Abstract Starts Here ###",
        "Ensure the abstract does not contain the research conclusion",
        "DO NOT END A SENTENCE WITH A COMMA",
        "Maximize the abstract to 300 words",
        "Ensure each sentence is around 2 to 3 citations at most.",
        "Avoid listing multiple authors sequentially as",
        "AVOID THESE IN THE ABSTRACT",
        "Write in formal and professional academic language",
        "(ABSTRACT SHOULD NOT BE WRITTEN IN FIRST PERSON)",
        "Ensure the abstract is comprehensive, covering all the background, method, and achievement BUT NOT EXCESSIVELY DETAILED.",
        "Do not copy and paste the introduction from the research paper into the abstract.",
        "Ensure that each sentence is concise, with no more than two to three citations.",
        "Limit the number of authors cited in a single sentence to one or two, using \"and others\" (et al.) for subsequent authors.",
        "as submitting a shorter abstract may not provide sufficient background information for reviewers to fully appreciate the research's significance.",
        "Please submit the research introduction as it appears in the paper.",
        "Do not list multiple authors consecutively.",
        "to simplify citations with multiple authors.",
        "* The abstract must be 300 words long. *",
        "Restrictions:",
        "Make sure each section starts with “Background of” or “Statement of,” “Method,” “Results,” “Discussion of,” or “Conclusion of.”",
        "If necessary, split the abstract into multiple sections.",
        "Do not exceed the 300-word limit.",
        "* Each sentence should be around 2 to 3 citations at most.",
        "* Cover every research aspect in great detail.",
        "Use a formal, academic language and maintain a third-person tone throughout.",
        "* Avoid listing multiple authors in a row. Instead, use “et al.” for simplicity.",
        "Ensure the abstract covers all research aspects in detail, including background, purpose or question, methodology or method, significant highlights of qualitative or quantitative results, and in-depth discussion or interpretation of the conclusions.",
        "Abstracts MUST be written in a professional and scholarly tone, avoiding first-person pronouns, contractions, and colloquial language.",
        "Please complete the abstract in a professional, academic tone, using third-person language and a formal voice.",
        "Research Background:",
        "ABSTR ACT END ###",
        "Footnote",
        "### ABSTRACT END ===",
        "Here is the research introduction.", 
        "Please process it to create a detailed and comprehensive abstract that meets the requirements mentioned above.",
        "Note: The abstract must be written in a formal academic tone, in third person, and at a minimum of 300 words but no more than 700 words.",
        "Each sentence should not exceed 2 to 3 citations.",
        "Limit consecutive citations to a single author to avoid listing multiple authors in a row.",
        "Instead, use “et al.” for simplicity.",
        "Research background, objectives, methods, findings, and significance must all be thoroughly covered in the abstract.",
        "Ensure the abstract meets the target word count and maintains a professional tone throughout.",
        "Good luck!",
        "If you have any further questions or concerns, please do not hesitate to contact me.",
        "Note:",
        "Kindly ensure the abstract is self-contained and does not require readers to refer to the main body of the paper for further information.",
        "Also, please note that the abstract should not contain any undefined acronyms, abbreviations, or technical jargon that may be unfamiliar to non-experts in the field.",
        "Furthermore, please maintain a formal and professional tone throughout the abstract, avoiding the use of first-person pronouns, contractions, and colloquial language.",
        "In addition, the abstract must provide a clear and concise overview of the entire paper, covering the research motivation, methodology, results, and implications.",
        "It is also essential to highlight the novelty and originality of the work, as well as the significance of the findings and their potential impact on the relevant field of study.",
        "Finally, please ensure that there are no grammatical errors, typos, or formatting issues in the abstract.",
        "I hope this helps. If you have any further questions or concerns, please do not hesitate to ask. I look forward to your response.",
        "Best regards, the AI assistant.",
        "Research Background and Motivation:", 
        "The generated abstract should include all the necessary information from the provided research introduction.",
        "Please do not add or omit any details.",
        "Here is the provided research background introduction.",
        "### ABSTRACT START ###",
        "### ABSTRACT END ###",
        "### ABDSTRACT END ###",
        "### Abstract Ends Here ###",
        "ABstract END ###",
        "###",
        "####",
        "a well-structured abstract that includes:",
        "- A thorough introduction to the research background (1-2 sentences)",
        "- Clearly states the research objective or question",
        "- Introduces the proposed method or solution",
        "- Highlights the main contributions or innovations",
        "- Reports the experimental results or performance evaluation",
        "- Concludes the abstract with a summary of the significance or impact of the research",
        "Background:",
        "### STATEMENT OF PURPOSE:",
        "### PLEASE GENERATE AN ABSTRACT THAT MEETS ALL THE ABOVE GUIDELINES AND IS COMPLETED IN A TIMELY FASHION.###",  
        "##### WE LOOK FORWARD TO REVIEWING YOUR WORK! #####  ##### GOOD LUCK WITH YOUR ABSTRACT GENERATION! #####",
        "##### WE HOPE YOU'LL MEET ALL THE REQUIREMENTS AND SUBMIT A HIGH-QUALITY ABSTRACT IN TIME.#####",
        "##### WE'RE HERE TO HELP! IF YOU NEED ANY ASSISTANCE OR HAVE ANY QUESTIONS, PLEASE DON'T HESITATE TO REACH OUT TO US!#####",
        "##### THANK YOU FOR YOUR UNDERSTANDING AND COOPERATION! WE'RE EXCITED TO SEE YOUR FINISHED ABSTRACT!#####",
        "####### END OF INSTRUCTIONS #######",
        "###### ABSTRACT GENERATED BY AI ASSISTANT ######",
        "###### PLEASE REVIEW AND EDIT CAREFULLY BEFORE SUBMITTING TO YOUR JOURNAL/PUBLISHER. ######",
        "WE ARE NOT RESPONSIBLE FOR THE QUALITY, ACCURACY, OR COMPLETENESS OF THE GENERATED ABSTRACT.",
        "PLEASE MAKE ANY NECESSARY CHANGES TO MEET THE SPECIFIC REQUIREMENTS OF YOUR SUBMISSION. THANK YOU! ####",
        "### ABSTRACT END ===  Limit: 300 words (approx.)   ###",
        "DO NOT INCREASE THE ABSTRACT LENGTH ###",
        "### MAKE SURE ALL SENTENCES ARE WELL STRUCTURED, CLEAR, AND PROFESSIONALLY WRITTEN IN FORMAL ACADEMIC ENGLISH (THIRD PERSON) ###",
        "### INCLUDE ALL THE RESEARCH BACKGROUNDS, OBJECTIVES, METHODS, FINDINGS, ANALYSES, AND CONTRIBUTIONS EXTENSIVELY, BUT DO NOT LIST MULTIPLE AUTHORS CONSECUTIVELY—USE \"ET AL.\" INSTEAD###",
        "### BE SURE TO COVER ALL ASPECTS OF THE INTRODUCTION THOROUGHLY, BUT AVOID ANY SUPERFLUOUS OR UNNECESSARY INFORMATION THAT MIGHT Inflate THE Abstract TO EXCEED THE 300-WORD LIMIT###",
        "### NO CITATION REFERENCES IN THE FIRST PARAGRAPH (BACKGROUND) ###",
        "### EACH SENTENCE SHOULD HAVE A MAXIMUM OF 3-4 CITATIONS (ET AL. IS ACCEPTABLE WHEN THERE ARE MORE THAN 4 AUTHORS)###",
        "##### ABSTRACT MUST BE COMPREHENSIVE, SUBSTANTIVE, AND ADEQUATELY COVERS THE ENTIRE RESEARCH INTRODUCTION IN EXTENSIVE DETAIL ######",
        "##### DO NOT SKIMP ON ANY ASPECT OF THE BACKGROUND, OBJECTIVE, METHOD, FINDING, ANALYSIS, OR CONTRIBUTION—ENSURE IT IS WELL-RESEARCHED, IN-DEPTH, AND COMPLETELY COVERED IN THE ABBREVIATED ABSTRACT ######",
        "##### THE WORD COUNT MUST NOT EXCEED 700 WORDS (APPROXIMATELY) #####",
        "##### KEEP THE SENTENCES CLEAR, CONCISE, AND WELL-STRUCTURED, AVOIDING ANY AMBIGUITY OR CONFUSION #####",
        "##### THE TONE MUST BE FORMAL, SCHOLARLY, AND IN THE THIRD PERSON (NO FIRST-PERSON REFERENCE)#####",
        "##### AVOID LISTING ALL THE AUTHORS SEQUENTIALLY IN THE CITATION (USE ET AL. FOR 3 OR MORE AUTHORS)#####",
        "##### ENSURE ALL STATEMENTS ARE FACTUALLY ACCURATE, VERIFIABLE, AND RELIABLY SOURCED FROM THE ORIGINAL RESEARCH PAPER (NO INACCURACIES OR FABRICATIONS ALLOWED)#####",
        "Important: Limit each sentence to a maximum of 2-3 citation references."
        "### SUBMIT YOUR ABSTRACT AS A SINGLE MICROSOFT WORD (.DOCX) FILE.",
        "### MAKE SURE TO FOLLOW THE ABOVE GUIDELINES EXACTLY.",
        "FAILURE TO DO SO MAY RESULT IN DELAYS OR REJECTION OF YOUR SUBMISSION.",
        "### GOOD LUCK WITH YOUR RESEARCH! ### ENJOY THE WRITING PROCESS!",
        "### PREPARE THOROUGHLY TO ENSURE A HIGH-QUALITY ABSTRACT.",
        "### WE'RE ROOTING FOR YOU! ### GOOD FORTUNE AWAITS!",
        "### WRITE WITH CONFIDENCE! ### SUCCESS IS WITHIN REACH! ###"
    ]
    
    for fragment in prompt_fragments:
        if fragment in abstract:
            pos = abstract.find(fragment) + len(fragment)
            abstract = abstract[pos:].strip()
    
    #確保摘要以完整句子開始
    if abstract and abstract[0].islower():
        pos = abstract.find(". ")
        if pos != -1 and pos + 2 < len(abstract):
            abstract = abstract[pos+2:].strip()
    
    #移除可能的引號
    abstract = abstract.strip('"\'')
    
    #修正引用鏈問題
    abstract = fix_citation_chains(abstract)
    
    #確保摘要完整
    abstract = ensure_complete_abstract(abstract)
    
    return abstract

#讀取train
with open("train.json", "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

#讀取test
with open("test.json", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

#將提示模板應用到 test_data
test_data_with_prompt = []
for item in test_data:
    prompt_text = PROMPT_TEMPLATE.format(introduction=item["introduction"])
    test_data_with_prompt.append({
        "paper_id": item["paper_id"],
        "introduction": prompt_text,
        "abstract": item.get("abstract", "")
    })

#設定模型  "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CHECKPOINT_DIR = f"./313706038/checkpoint-{MODEL_NAME}"  #檢查點保存路徑

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

#載入tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  #設置填充標記

# load_path = MODEL_NAME
# load_from_checkpoint = False
# if os.path.exists(CHECKPOINT_DIR):
#     print(f"Loading model checkpoint from {CHECKPOINT_DIR}")
#     load_path = CHECKPOINT_DIR
#     load_from_checkpoint = True
# else:
#     print(f"No checkpoint found at {CHECKPOINT_DIR}. Loading base model {MODEL_NAME}.")
#     # 注意：如果沒有檢查點，就不應該跳過訓練

# # 統一載入
# model = AutoLigerKernelForCausalLM.from_pretrained(
#     load_path,
#     quantization_config=bnb_config if not load_from_checkpoint else None, # 如果從檢查點載入，通常不需要再指定量化，除非檢查點未量化
#     device_map="auto",
#     attn_implementation="flash_attention_2", # 始終嘗試使用 Flash Attention
#     torch_dtype=torch.float16 # 配合 bnb_config 或 FP16 訓練
# )

# if not load_from_checkpoint:
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=16, # 嘗試 32 或 64
#         lora_alpha=32, # 相應調整為 64 或 128
#         lora_dropout=0.1,
#         target_modules=[
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ]
#     )
#     model = get_peft_model(model, lora_config)
#     print("Applied new LoRA configuration to the base model.")
# else:
#     # 如果從檢查點加載，PEFT 會自動處理 LoRA 層
#     print("Loaded model with LoRA layers from checkpoint.")

if os.path.exists(CHECKPOINT_DIR):
    print(f"Loading model checkpoint from {CHECKPOINT_DIR}")
    
    model = AutoLigerKernelForCausalLM.from_pretrained(
        CHECKPOINT_DIR,
        quantization_config=bnb_config,
        device_map="auto"
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     CHECKPOINT_DIR,
    #     quantization_config=bnb_config,
    #     device_map="auto"
    # )
else:
    print(f"No checkpoint found at {CHECKPOINT_DIR}. Loading base model.")

    model = AutoLigerKernelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     quantization_config=bnb_config,
    #     device_map="auto"
    # )

model = AutoLigerKernelForCausalLM.from_pretrained(
    MODEL_NAME,
    # attn_implementation="eager",  #使用 eager 注意力實現
    attn_implementation="flash_attention_2",
    load_in_4bit=True,            #4-bit 量化
    device_map="auto"             #自動分配到 GPU
)


# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     # attn_implementation="eager",  #使用 eager 注意力實現
#     attn_implementation="flash",
#     load_in_4bit=True,            #4-bit 量化
#     device_map="auto"             #自動分配到 GPU
# )

# 設定LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])

model = get_peft_model(model, lora_config)

#Tokenize資料（批次處理）
def tokenize_function(examples):
    inputs = [
        PROMPT_TEMPLATE.format(introduction=intro) + abstr + "\n### ABSTRACT END ###"
        for intro, abstr in zip(examples["introduction"], examples["abstract"])
    ]
    tokenized_inputs = [tokenizer.encode(input_text, add_special_tokens=True) for input_text in inputs]
    max_intro_len = max(len(tokenizer.encode(item["introduction"])) for item in train_data)
    max_abs_len = max(len(tokenizer.encode(item["abstract"])) for item in train_data)
    dynamic_max_length = min(max(max_intro_len, max_abs_len) + 100, 1500)  # 留緩衝空間
    # dynamic_max_length = min(max(max(len(tokens) for tokens in tokenized_inputs), 512), 600)
    
    tokenized = tokenizer(
        inputs,
        max_length=2048,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"]
    }

#建立 Dataset
dataset = Dataset.from_list(train_data)
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# # 建立 Dataset
# dataset = Dataset.from_list(train_data)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# train_val_split = tokenized_datasets.train_test_split(test_size=0.2)
# train_dataset = train_val_split["train"]
# val_dataset = train_val_split["test"]

#設定訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    num_train_epochs=7,
    save_strategy="epoch",
    save_total_limit=1,
    weight_decay=0.01,
    report_to="none",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
    logging_first_step=True,
    fp16=True,
    load_best_model_at_end=True,  
    metric_for_best_model="eval_loss",  
    greater_is_better=False, 
)

#分割訓練資料
train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

#定義Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

#開始訓練
trainer.train()

#如果沒有檢查點，則進行訓練並保存最終檢查點
if not os.path.exists(CHECKPOINT_DIR):
    print("Training model from scratch...")
    trainer.train()
    #保存最終模型檢查點
    trainer.save_model(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"Model checkpoint saved to {CHECKPOINT_DIR}")
else:
    print("Skipping training as checkpoint already exists.")

#生成測試集摘要
predictions = []
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


for sample in test_data_with_prompt:
    prompt = sample["introduction"]
    original_intro = test_data[test_data_with_prompt.index(sample)].get("introduction", "")
    
    intro_tokens = len(tokenizer.encode(original_intro, add_special_tokens=True))
    target_word_count = min(300 + (intro_tokens // 2), 400)  
    target_tokens = max(300, min(int(intro_tokens * 1.5), 400))  # 根據introduction長度調整
    target_tokens = int(target_word_count * 1.5)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False, max_length=3000)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=6,
        length_penalty=1.5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.7,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id                   
    )
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    abstract = extract_abstract(full_output, prompt)
    abstract = postprocess_abstract(abstract, sample["paper_id"], original_intro, target_word_count)

    
    #保存結果
    predictions.append({"paper_id": sample["paper_id"], "abstract": abstract})
    print(f"生成摘要：{sample['paper_id']}")
    # print(predictions)

#保存結果
with open("313706038_0406_2.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"成功生成{len(predictions)} 個摘要")
