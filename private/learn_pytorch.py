'''
loss.backward()
optimizer.step()
optimizer.zero_grad()
every neural network
using raw math and basic arrays
'''
'''
data structures needed
direct creation from data -> torch.tensor
'''
import torch

data = [[1,2,3],[4,5,6]]
my_tensor = torch.tensor(data)
print(my_tensor)

'''creation from a desired shape'''
shape =(2,3)

ones = torch.ones(shape)
zeros = torch.zeros(shape)
random_tensor = torch.randn(shape)
print(ones)
print(zeros)
print(random_tensor)

'''creation by mimicking another tensor'''
template = torch.tensor([[1,2,3],[4,5,6]])
rand_like = torch.randn_like(template, dtype=torch.float32)

print(template)
print(rand_like)

# shape type and device
tensor = torch.randn((2,3))
print(tensor.shape)
print(tensor.dtype)
print(tensor.item())
print(tensor.device)

"autagrad built in calculator"
'''requires grad = true'''


