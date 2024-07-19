import pandas as pd
diagnosis_path = 'data/DIAGNOSES_ICD.csv'
event_path = 'data/NOTEEVENTS.csv'
procedures_path = 'data/PROCEDURES_ICD.csv'


dia_df = pd.read_csv(diagnosis_path)
event_df = pd.read_csv(event_path)
proced_df = pd.read_csv(procedures_path)

discharge_df = event_df[event_df['CATEGORY'] == 'Discharge summary']
icd_data = pd.concat([dia_df, proced_df])
merged_data = discharge_df.merge(icd_data[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']],
                                 on=['SUBJECT_ID', 'HADM_ID'],
                                 how='left')

group_df = merged_data.groupby(['SUBJECT_ID', 'HADM_ID', 'TEXT'])['ICD9_CODE'].agg(list).reset_index()
group_df.to_csv('data/group_df.csv')