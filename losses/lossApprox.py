import torch

# Set eta
eta = torch.randn(1,)
print(eta)

# Netowork out and labels
outputs = torch.randn(5,2)
print(outputs)
c_l = torch.tensor([[1, 0, 1, 0, 1]])

def loss1(outputs, c_l):
    sm = torch.nn.functional.softmax(outputs, dim=1)
    loss1 = torch.exp(-eta)*torch.log(sm[torch.arange(len(sm),dtype=torch.long), (c_l.squeeze()).long()])
    num = torch.sum(torch.exp(torch.exp(-eta)*outputs), dim=1)
    den = (torch.sum(torch.exp(outputs), dim=1))** torch.exp(-eta)
    loss1 -= torch.log(num/den)
    return loss1

def loss2(outputs, c_l):
    return torch.exp(-eta)*outputs[torch.arange(len(outputs),dtype=torch.long), (c_l.squeeze()).long()] - torch.log(torch.sum(torch.exp(torch.exp(-eta)*outputs),dim=1))

print(loss1(outputs, c_l))
print(loss2(outputs, c_l))