
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix


class MultiLabel_Encoding():
    def __init__(self, all_labels) -> None:
        self.all_labels = all_labels
        self.mlb = MultiLabelBinarizer(classes=all_labels, sparse_output=False)
        # self.csr = None

    def fit_transform(self, labels):
        encoded_labels = self.mlb.fit_transform(labels)     
        # self.csr = csr_matrix(encoded_labels)
        return encoded_labels
    
    
    def add_transform(self, df, target_column, end_point_column):
        labels = df[target_column].tolist()

        encoded_label = self.fit_transform(labels=labels)
        df[end_point_column] = pd.Series([x for x in encoded_label], index=df.index)
        
        return df
