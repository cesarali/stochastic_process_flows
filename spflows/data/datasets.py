from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data[idx]
        context = series[:self.context_length]
        prediction = series[self.context_length:self.context_length + self.prediction_length]
        return {"context": context, "prediction": prediction}
