import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from cfg import CFG


class TrainDataset(Dataset):

    def __init__(self, df, tokenizer):
        df.reset_index(inplace=True)
        self.labels = df[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].values
        self.texts = df['full_text'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        output_ids = self.tokenizer(text,
                                    padding='max_length', max_length=CFG.max_position_embeddings, truncation=True)
        return {'input_ids': torch.as_tensor(output_ids['input_ids'], dtype=torch.long),
                'attention_mask': torch.as_tensor(output_ids['attention_mask'], dtype=torch.long),
                'labels': torch.as_tensor(label, dtype=torch.float)}



if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
    df = pd.read_csv('train.csv')
    train_dataset = TrainDataset(df, tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test = next(iter(train_loader))
    for key, value in test.items():
        print(key)
        print(value.shape)
    print('done')
