import pandas as pd
from typing import Dict, List, Any


class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['data']
    
    def load_csv(self, csv_path: str = None) -> pd.DataFrame:
        path = csv_path or self.config['input_csv']
        df = pd.read_csv(path)
        df = df.dropna(subset=self.config['text_columns'])
        return df
    
    def extract_text(self, row: pd.Series) -> str:
        parts = [str(row[col]) for col in self.config['text_columns'] if pd.notna(row[col])]
        return ' '.join(parts)
    
    def prepare_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        records = []
        for _, row in df.iterrows():
            records.append({
                'id': row[self.config['id_column']],
                'text': self.extract_text(row),
                'raw': row.to_dict()
            })
        return records
