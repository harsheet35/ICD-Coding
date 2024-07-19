
from utils.MultilabelEncoding import MultiLabel_Encoding
from utils.pass_prompts import process_notes
import pandas as pd
from typing import List




import re
def clear_icd_codes(codes: List) -> List:
    text = ""
    for code_part in codes:
        text += str(code_part)
    regex = r'\b[A-Z]?\d{2,4}(?:\.\d{1,4})\b'
    icd_codes = re.findall(regex, text)
    return icd_codes



def get_icd_codes_from_gpt(data: pd.DataFrame, result_file_path: str, candidate_list: str, all_icd_codes: List) -> None:
    
    data.reset_index()
    print("FUnction called")
    results, _ = process_notes(data=data, candidate_list=candidate_list, test_rows=len(data))

    dic_top_50 ={'SUBJECT_ID': [res[0] for res in results],
             'HADM_ID': [res[1] for res in results],
             'TEXT': [res[2] for res in results],
             'ICD9_CODE': [res[3] for res in results]}
    
    res_df = pd.DataFrame(dic_top_50)
    res_df['ICD9_CODE'] = res_df['ICD9_CODE'].apply(clear_icd_codes)

    icd_encoding = MultiLabel_Encoding(all_labels=all_icd_codes)

    icd_encoding.add_transform(df=res_df, target_column='ICD9_CODE', end_point_column='encoded_ICD9_CODES')

    res_df.to_csv(result_file_path, mode='a', index=False, header=True)


def process_dataframe_in_chunks(dataframe: pd.DataFrame, chunk: int, candidate_list: str, result_file_path: str, ICD_codes: List):
    
    for i in range(chunk):
        start_idx = i * chunk
        end_idx = min(start_idx + chunk, len(dataframe))
        data = dataframe.iloc[start_idx:end_idx]
        print("Processing from start index", start_idx)
        print("\nProcessing from end Index", end_idx)
        print("\n---------------------------------------------------------------\n")
        try:
            get_icd_codes_from_gpt(data=data, result_file_path=result_file_path, candidate_list=candidate_list, all_icd_codes=ICD_codes)
        except Exception as e:
            print(e)
            break

def load_and_process_dataframe_in_chunks(file_path:str,chunksize:int, candidate_list: str, result_file_path: str, ICD_codes: List):
    header_written = False
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        try:
            chunk = chunk.reset_index()
            results, _ = process_notes(data=chunk, candidate_list=candidate_list, test_rows=len(chunk))
            dic_top_50 ={'SUBJECT_ID': [res[0] for res in results],
                    'HADM_ID': [res[1] for res in results],
                    'TEXT': [res[2] for res in results],
                    'ICD9_CODE': [res[3] for res in results]}
                
            res_df = pd.DataFrame(dic_top_50)
            res_df['ICD9_CODE'] = res_df['ICD9_CODE'].apply(clear_icd_codes)

            icd_encoding = MultiLabel_Encoding(all_labels=ICD_codes)
            icd_encoding.add_transform(df=res_df, target_column='ICD9_CODE', end_point_column='encoded_ICD9_CODES')
            res_df.to_csv(result_file_path, mode='a', index=False, header=not header_written)
            header_written = True
            print(f'loop {n} completed')
            n += 1
        except Exception as e:
            print('\n\n\n', e)
            break
        
    
