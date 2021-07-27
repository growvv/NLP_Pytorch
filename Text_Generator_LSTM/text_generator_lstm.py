import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get characters from string.printable
all_characters = string.printable
n_characters = len(all_characters)

# Read large file
file = unidecode.unidecode(open("data/names.txt").read())

class RNN(nn.Module):
    def __init__(self, input_size, seq_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * seq_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))  # (batch_size, seq_length * hidden_size)
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell

class Generator:
    def __init__(self) -> None:
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 2
        self.print_every = 100
        self.seq_size = 1
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for idx in range(len(string)):
            tensor[idx] = all_characters.index(string[idx])
        return tensor  # 下标组成的tensor

    def get_random_batch(self):
        # print(len(file))  # file 是一个超长的字符串
        # print(type(file))
        start_idx = random.randint(0, len(file)-self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1 # ?
        text_str = file[start_idx: end_idx]  # 去字符串中的256+1个字符
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_output = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            print("前面")
            print(type(text_str[:-1]))
            print(text_str[:-1][0])
            print(text_str[:-1][1])
            print("后面")
            text_input[i, :] = self.char_tensor(text_str[:-1]) # 前n个, text_str长n+1个
            text_output[i, :] = self.char_tensor(text_str[1:])  # 后n个

        return text_input.long(), text_output.long()

    def generate(self, initial_str="A", predict_len=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str)-1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)

        print("init:", initial_str)
        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        print("pred:", predicted)
        return predicted

    def train(self):
        self.rnn = RNN(n_characters, self.seq_size, self.hidden_size, self.num_layers, n_characters).to(device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f"run/names")

        print("=> String training")

        for epoch in range(self.num_epochs):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len
            
            if epoch %  self.print_every == 0:
                print(f"Loss: {loss}")
                # print(self.generate())
                self.generate()

            writer.add_scalar("Training loss", loss, global_step=epoch)

    

if __name__ == "__main__":
    gen_names = Generator()
    # gen_names.train()
    print(gen_names.char_tensor("hello"))
