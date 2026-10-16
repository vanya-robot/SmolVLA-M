import torch

from memory import CompressedMemory


batch_size = 2
history_len = 8
input_dim = 32
memory_dim = 128

model = CompressedMemory(input_dim=input_dim, memory_dim=memory_dim)
history = torch.randn(batch_size, history_len, input_dim)
memory = model(history)

assert memory.shape == (batch_size, memory_dim)

loss = memory.square().mean()
loss.backward()

print("history:", tuple(history.shape))
print("memory:", tuple(memory.shape))
print("backward: ok")
