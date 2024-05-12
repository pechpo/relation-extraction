"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""

from Exp3_DataSet import TextDataSet, TestDataSet
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN
import torch
import pickle
from tqdm import tqdm
from matplotlib import pyplot

def train(model, loader, loss_fn, opt):
    print("Start training......")
    num_batches = len(loader)
    model.train()
    with tqdm(total=num_batches) as pbar:
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            pbar.set_postfix({"loss":loss.item()})
            pbar.update(1)

            # Backpropagation
            loss.backward()
            opt.step()
            opt.zero_grad()

def validation(model, loader, loss_fn):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad(), tqdm(total=num_batches) as pbar:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            pbar.set_postfix({"loss":loss.item()})
            pbar.update(1)
            val_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    val_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss, correct


def predict(model, loader):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.eval()
    with torch.no_grad(), tqdm(total=num_batches) as pbar, open("exp3_predict_labels.txt", "w") as f:
        for X in loader:
            X = X.to(device)
            pred = model(X)
            pbar.update(1)
            for i in range(X.shape[0]):
                print(pred[i].argmax(0).item(), file=f)


if __name__ == "__main__":

    test_mode = True

    # 训练集验证集
    train_dataset = TextDataSet(filepath="./data/data_train_processed.pkl")
    train_loader = DataLoader(dataset=train_dataset, batch_size=16)

    val_dataset = TextDataSet(filepath="./data/data_val_processed.pkl")
    val_loader = DataLoader(dataset=val_dataset, batch_size=16)

    # 测试集数据集和加载器
    test_dataset = TestDataSet("./data/test_exp3_processed.pkl")
    test_loader = DataLoader(dataset=test_dataset, batch_size=16)

    print("Loading word embedding file......")

    with open("word_embedding.pkl", "rb") as f:
        a = pickle.load(f)

    # 初始化模型对象
    Text_Model = TextCNN(num_classes=44, num_embeddings=379861, embedding_dim=300
                        , embeddings_pretrained=torch.tensor(a["matrix"], dtype=torch.float32))
    device = ( "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = Text_Model.to(device)

    if not test_mode:

        # 损失函数设置
        loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
        # 优化器设置
        optimizer = torch.optim.Adam(params=Text_Model.parameters())  # torch.optim中的优化器进行挑选，并进行参数设置

        # 训练和验证
        epoch = 10
        current_best = 0
        x = []
        loss_l = []
        acc_l = []
        for i in range(epoch):
            train(model, loader=train_loader, loss_fn=loss_function, opt=optimizer)
            loss, acc = validation(model, loader=val_loader, loss_fn=loss_function)
            x.append(i)
            loss_l.append(loss)
            acc_l.append(acc)
            if acc > current_best:
                print("Current best, saving model......")
                torch.save(model.state_dict(), "model.pth")
                current_best = acc
        
        fig, ax = pyplot.subplots()
        ax.plot(x, loss_l, label="loss")
        ax.plot(x, acc_l, label="accuracy")
        ax.legend()
        pyplot.savefig('training_result.png', dpi = 300)
    else:
        model.load_state_dict(torch.load("model.pth"))

    # 预测（测试）
    predict(model, test_loader)
