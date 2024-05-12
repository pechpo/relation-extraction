import torch
from torch.utils.data import Dataset
import pickle

# 训练集和验证集
class TextDataSet(Dataset):
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        #  dict_keys(['word1', 'word2', 'sentence', 'label'])
        word1 = data["word1"]
        word2 = data["word2"]
        sentence = data["sentence"]
        self.data = torch.zeros((len(word1), 200), dtype=torch.int)
        for i in range(len(word1)):
            self.data[i][0] = word1[i]
            self.data[i][1] = word2[i]
            tmp = torch.tensor(sentence[i], dtype=torch.int)
            self.data[i][2:tmp.shape[0]+2] = torch.tensor(sentence[i])
        self.label = data["label"]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(Dataset):
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        #  dict_keys(['word1', 'word2', 'sentence', 'label'])
        word1 = data["word1"]
        word2 = data["word2"]
        sentence = data["sentence"]
        self.data = torch.zeros((len(word1), 200), dtype=torch.int)
        for i in range(len(word1)):
            self.data[i][0] = word1[i]
            self.data[i][1] = word2[i]
            tmp = torch.tensor(sentence[i])
            self.data[i][2:tmp.shape[0]+2] = torch.tensor(sentence[i], dtype=torch.int)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    trainset = TextDataSet(filepath="./data/data_train_processed.pkl")
    testset = TestDataSet(filepath="./data/test_exp3_processed.pkl")
    print("训练集长度为：", len(trainset))
    print("测试集长度为：", len(testset))
    #max_len = 0
    #for i in range(trainset.__len__()):
        #max_len = max(max_len, len(trainset.__getitem__(i)[0]))
    #for i in range(testset.__len__()):
        #max_len = max(max_len, len(testset.__getitem__(i)))
    #print(f"最大句子长度={max_len}")