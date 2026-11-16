from torch.utils.data import Dataset, DataLoader

class Dataset_real(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        #print("item: ", item)
        return item
    
    
    def __len__(self):
        return len(self.dataset)
