# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=64, num_layers=1):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # 可以根据任务需求选择使用RNN或者平均池化层
        self.rnn = nn.RNN(vector_dim, hidden_size, num_layers, batch_first=True)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        # self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        # 使用RNN的情况
        # output, _ = self.rnn(x)
        # x = output[:, -1, :]  # 或者写成 x = _.squeeze()
        # 使用平均池化层的情况
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()

        y_pred = self.classify(x)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 找到字符 "a" 在 x 中的位置，如果不存在，设为默认值 -1
    a_position = next((i for i, char in enumerate(x) if char == 'a'), 6)
    y = a_position
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print("本次预测集中共有 %d 个样本" % (len(y)))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        predicted_classes = torch.argmax(y_pred, dim=1)

        for y_p, y_t in zip(predicted_classes, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
        return accuracy

def main():
    epoch_num = 40
    batch_size = 20
    train_sample = 1000
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.001
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length + 1)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "practice.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))

if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "azsdfg", "rawdeg", "nkawww", "zsdafg", "rwdeag", "nkwwwa"]
    predict("practice.pth", "vocab.json", test_strings)
