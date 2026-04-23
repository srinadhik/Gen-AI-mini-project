import torch
import torch.nn as nn

text = "hello world hello ai"
chars = list(set(text))
char2idx = {ch:i for i,ch in enumerate(chars)}
idx2char = {i:ch for ch,i in char2idx.items()}

data = [char2idx[ch] for ch in text]

class LSTMGen(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 10)
        self.lstm = nn.LSTM(10, 20)
        self.fc = nn.Linear(20, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

model = LSTMGen(len(chars))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = 0
    hidden = None
    for i in range(len(data)-1):
        inp = torch.tensor([[data[i]]])
        target = torch.tensor([data[i+1]])

        out, hidden = model(inp, hidden)
        loss = loss_fn(out.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss}")

# Generate text
inp = torch.tensor([[data[0]]])
hidden = None
result = ""

for _ in range(20):
    out, hidden = model(inp, hidden)
    prob = torch.softmax(out.squeeze(), dim=0)
    idx = torch.argmax(prob).item()
    result += idx2char[idx]
    inp = torch.tensor([[idx]])

print("Generated text:", result)
