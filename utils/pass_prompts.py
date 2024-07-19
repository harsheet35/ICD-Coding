
import openai
import pandas as pd
import tiktoken  # to count tokens
from dotenv import load_dotenv
import os
import nltk
from nltk.tokenize import sent_tokenize


load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')



def split_document_into_segments(doc, num_segments):
    """
    Split a document into multiple segments with approximately equal number of sentences.

    Args:
        doc (str): The document text to be split.
        num_segments (int): The number of segments to split the document into.

    Returns:
        List[str]: A list of text segments, each containing approximately equal number of sentences.
    """
   
    nltk.download('punkt')
    
    sentences = sent_tokenize(doc)
    
    
    num_sentences = len(sentences)
    sentences_per_segment = num_sentences // num_segments
    remainder = num_sentences % num_segments
    
    segments = []
    start_idx = 0
    
    for i in range(num_segments):
        end_idx = start_idx + sentences_per_segment
        
        if remainder > 0:
            end_idx += 1
            remainder -= 1
        
        segment = ' '.join(sentences[start_idx:end_idx])
        segments.append(segment)
        start_idx = end_idx
    
    return segments





def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)

def prepare_prompt(note, candidates):
    return f"""
    As a proficient clinical coding professional, it is your responsibility to assign ICD 9 codes given the CLINICAL NOTE from the CANDIDATE LIST provided below.
    —
    CLINICAL NOTE (or partial):
    {note}
    —
    Here is a CANDIDATE LIST of 50 ICD 9 codes and their associated descriptions to assign:
    {candidates}
    —
    For each disease/procedure based on the context in CLINICAL NOTE, you must generate a list of strings containing the ICD 9 codes you assigned.
    I repeat put all the Codes in a list of string don't generate any extra sentences.
    """

candidate_list = """
    V70.0 - Routine Medical Exam
    V76.2 - Screening for Malignant Neoplasm of Cervix
    401.1 - Benign Hypertension
    V76.44 - Screening for Malignant Neoplasm of Prostate
    250.00 - Diabetes Mellitus without Complications, Type II or Unspecified Type
    401.9 - Hypertension, Unspecified
    780.79 - Other Malaise and Fatigue
    599.0 - Urinary Tract Infection, Site Not Specified
    272.4 - Other and Unspecified Hyperlipidemia
    V72.60 - Laboratory Examination, Unspecified
    244.9 - Hypothyroidism, Unspecified
    V58.69 - Long-term (current) Use of Other Medications
    788.1 - Dysuria
    272.2 - Mixed Hyperlipidemia
    272.0 - Pure Hypercholesterolemia
    268.9 - Unspecified Vitamin D Deficiency
    462 - Acute Pharyngitis
    250.02 - Diabetes Mellitus without Complications, Type II or Unspecified Type, Uncontrolled
    V58.61 - Long-term (current) Use of Anticoagulants
    789.00 - Abdominal Pain, Unspecified Site
    285.9 - Anemia, Unspecified
    427.31 - Atrial Fibrillation
    599.70 - Hematuria, Unspecified
    787.91 - Diarrhea
    780.60 - Fever, Unspecified
    V69.2 - High Risk Sexual Behavior
    V72.62 - Laboratory Examination, General Medical Examination
    790.6 - Abnormal Blood Chemistry
    595.0 - Acute Cystitis
    257.2 - Testicular Hypofunction
    244.8 - Acquired Hypothyroidism
    616.10 - Vaginitis and Vulvitis, Unspecified
    V20.2 - Routine Child Health Examination
    786.2 - Cough
    790.29 - Other Abnormal Glucose
    530.81 - Esophageal Reflux
    V73.88 - Screening for Chlamydial Disease
    782.3 - Edema
    V77.91 - Screening for Lipoid Disorders
    780.4 - Dizziness and Giddiness
    V72.31 - Routine Gynecological Examination
    V22.1 - Supervision of Other Normal Pregnancy
    280.9 - Iron Deficiency Anemia, Unspecified
    300.00 - Anxiety State, Unspecified
    682.9 - Cellulitis and Abscess, Unspecified
    789.07 - Abdominal Pain, Generalized
    719.40 - Pain in Joint, Unspecified
    788.41 - Urinary Frequency
    784.0 - Headache
    783.1 - Abnormal Weight Gain
"""

candidate_dict = {}

for line in candidate_list.strip().split('\n'):
    code, description = line.split(' - ')
    candidate_dict[code] = description




def prepare_prompt_se(note, candidates):

    candidates = eval(candidates)

    results = [(candidate, candidate_dict[candidate]) for candidate in candidate_dict.keys()]

    return f"""
    As a proficient clinical coding profes-
    sional, it is your responsibility to extract evidence
    when assigning ICD code. Given the list of ICD 9
    CANDIDATE codes (diseases/procedures) to assign,
    you need to verify each code by extracting associated
    evidence sentence from CLINICAL NOTE. You could
    inference based on basic medical commonsense, such
    as prescription of metformin is evidence to type 2 di-
    abetes.
    —
    ICD 9 CANDIDATE codes and descriptions: {results}.
    —
    Here is the CLINICAL NOTE split by sentence,
    : {note}

    
    —
    When assigning ICD code, you should:
    1. Carefully assign ICD code to each sentence as
    evidence even ICD code is already assigned in the pre-
    vious sentence;
    2. If multiple ICD code found in one sentence, label
    them all and separate them by semicolon;
    3. Do not assign ICD code if it is negated or ruled
    out in the CLINICAL NOTE, for example you should
    not assign ”287.5” if ”No leukemia or thrombocytope-
    nia”;
    4. Include ICD code only, not the associated En-
    glish description.


    you need to return the output format as a list of tuple 
    containing the sentence that lead to that ICD code and the ICD code
    like this [(Sentence1, ICD_CODE1), (Sentence2, ICD_CODE1), (Sentence2, ICD_CODE2)].
    Give only the output and not anything extra
    """














def split_text(text, max_tokens, model="gpt-3.5-turbo"):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    split_texts = []
    
    if len(tokens) > max_tokens:
        start = 0
        while start < len(tokens):
            end = min(len(tokens),start + max_tokens)
            start_strip = max(0, start - 10) #Overlapping to retain context
            split_texts.append(enc.decode(tokens[start_strip:end]))
            start = end
    else:
        split_texts.append(text)
    
    return split_texts

def process_notes(data, candidate_list = candidate_list, model="gpt-3.5-turbo", max_tokens=4096, test_rows=4):
    print('Subject ID:', data.iloc[0,0])
    results = []
    total_tokens = 0
    # print(data.head(1))
    for index, row in data.iterrows():
        print(index)
        if index > test_rows:
            break
        
        note = row['TEXT']
        subject_id = row['SUBJECT_ID']
        hadm_id = row['HADM_ID']
        
        prompt = prepare_prompt(note, candidate_list)
        token_count = count_tokens(prompt, model)
        
        if token_count > max_tokens:
            split_notes = split_text(note, max_tokens - count_tokens(prepare_prompt("", candidate_list), model), model)
        else:
            split_notes = [note]
        
        icd_code_per_id = []
        for part in split_notes:
            prompt = prepare_prompt(part, candidate_list)
            token_count = count_tokens(prompt, model)
            total_tokens += token_count
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a clinical coding professional."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024
            )
            
            icd_codes = response.choices[0].message.content.strip()
            icd_code_per_id.append(icd_codes)

            # print(icd_codes)
        results.append((subject_id, hadm_id, note,  icd_code_per_id))
    # print(results)   
    return results, total_tokens

# results, total_tokens = process_notes(data, candidate_list)

# for res in results:
#     print('\n\n\n',res)
# print(f"Total tokens: {total_tokens}")




def process_notes_se(data, model="gpt-3.5-turbo", max_tokens=4096, test_rows=4):
    print('Subject ID:', data.iloc[0,0])
    results = []
    total_tokens = 0
    # print(data.head(1))
    for index, row in data.iterrows():
        print(index)
        if index > test_rows:
            break
        
        note = row['TEXT']
        subject_id = row['SUBJECT_ID']
        hadm_id = row['HADM_ID']
        icd_code = row['ICD9_CODE']
        
        prompt = prepare_prompt_se(note, icd_code)
        token_count = count_tokens(prompt, model)
        
        if token_count > max_tokens:
            split_notes = split_text(note, max_tokens - count_tokens(prepare_prompt_se("", icd_code), model), model)
        else:
            split_notes = [note]
        
        icd_code_per_id = []
        for part in split_notes:
            prompt = prepare_prompt_se(part, icd_code)
            token_count = count_tokens(prompt, model)
            total_tokens += token_count
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a clinical coding professional."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024
            )
            
            icd_codes = response.choices[0].message.content.strip()
            icd_code_per_id.append(icd_codes)

            # print(icd_codes)
        results.append((subject_id, hadm_id, note,  icd_code_per_id))
    # print(results)   
    return results, total_tokens
