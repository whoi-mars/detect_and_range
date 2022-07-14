import torch
import torch.nn as nn
import warnings

class SelectiveMSEAndClass(nn.Module):
    
    """
    Class for loss which adds binary cross entropy loss and MSE loss
    for range predictions. The MSE loss only penalizes incorrect range
    predictions for examples of class 1 (with a call in the spectrogram).

    Parameters
    ----------
    alpha: float, weight coefficient for the range loss term.
    """
    
    def __init__(self, alpha=1):
        super().__init__()
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCELoss()
        self.alpha = alpha
        self.log_vars = None # For ease of saving loss state
        
    def forward(self, outputs, r_labels, c_labels):
        
        # Isolate ranges for call-containing example only
        call_outs = outputs[c_labels.squeeze() == 1, 0]
        call_r_labels = r_labels[c_labels.squeeze() == 1]
        
        # Ignore MSE loss entirely if there are no calls in the batch for some reason.
        # In practice this shouldn't really happen because 
        # there are an equal number of noise/call examples.
        if len(call_outs) == 0:
            c_loss = self.BCE(outputs[:,1].squeeze(), c_labels.squeeze())
            r_loss = 0
            warnings.warn("No calls-containing examples in the batch.")
        else:
            c_loss = self.BCE(outputs[:,1].squeeze(), c_labels.squeeze())
            r_loss = self.alpha*self.MSE(call_outs.squeeze(), call_r_labels.squeeze())

        return r_loss + c_loss, r_loss, c_loss

class SelectiveSSEAndClass(nn.Module):
    
    """
    Class for loss which adds binary cross entropy loss and SSE loss
    for range predictions. The SSE loss only penalizes incorrect range
    predictions for examples of class 1 (with a call in the spectrogram).

    Parameters
    ----------
    alpha: float, weight coefficient for the range loss term.
    """
    
    def __init__(self, alpha=1):
        super().__init__()
        self.MSE = nn.MSELoss(reduction='sum')
        self.BCE = nn.BCELoss()
        self.alpha = alpha
        self.log_vars = None # For ease of saving loss state
        
    def forward(self, outputs, r_labels, c_labels):
        
        # Isolate ranges for call-containing example only
        call_outs = outputs[c_labels.squeeze() == 1, 0]
        call_r_labels = r_labels[c_labels.squeeze() == 1]
        
        # Ignore MSE loss entirely if there are no calls in the batch for some reason.
        # In practice this shouldn't really happen because 
        # there are an equal number of noise/call examples.
        if len(call_outs) == 0:
            c_loss = self.BCE(outputs[:,1].squeeze(), c_labels.squeeze())
            r_loss = 0
            warnings.warn("No calls-containing examples in the batch.")
        else:
            c_loss = self.BCE(outputs[:,1].squeeze(), c_labels.squeeze())
            r_loss = self.alpha*self.MSE(call_outs.squeeze(), call_r_labels.squeeze())

        return r_loss + c_loss, r_loss, c_loss

class UncertainSelectiveMSEAndClass(nn.Module):
    
    """
    Class for loss which adds binary cross entropy loss and MSE loss
    for range predictions. The MSE loss only penalizes incorrect range
    predictions for examples of class 1 (with a call in the spectrogram).

    This loss also implements the approach from Kendall et al. (https://arxiv.org/abs/1705.07115) 
    to learn weights for the classification and ranging tasks. 
    """
    
    def __init__(self, device, log_var_list=None):
        super().__init__()
        self.MSE = nn.MSELoss(reduction='none')

        # Learned variables for weighting the tasks
        if log_var_list is None:
            self.log_vars = [torch.tensor(0., device=device, requires_grad=True), torch.tensor(0., device=device, requires_grad=True)]
        else:
            self.log_vars = [torch.tensor(log_var_list[0].item(), device=device, dtype=torch.float32, requires_grad=True), torch.tensor(log_var_list[1].item(), device=device, dtype=torch.float32, requires_grad=True)]

    def forward(self, outputs, r_labels, c_labels):

        # Isolate ranges for call-containing example only
        call_outs = outputs[c_labels.squeeze() == 1, 0]
        call_r_labels = r_labels[c_labels.squeeze() == 1]

        # Get classification loss
        c_loss = torch.exp(-self.log_vars[1])*outputs[torch.arange(len(outputs),dtype=torch.long), (c_labels.squeeze()+1).long()] - torch.log(torch.sum(torch.exp(torch.exp(-self.log_vars[1])*outputs[:,1:]),dim=1))
    
        # Calculate range loss
        r_loss = self.MSE(call_outs.squeeze(), call_r_labels.squeeze())
        r_precision = torch.exp(-self.log_vars[0])
        r_loss = 0.5*(r_precision*r_loss + self.log_vars[0])
        r_loss_final = torch.zeros_like(c_loss) 
        r_loss_final[c_labels.squeeze() == 1] = r_loss

        # Get means
        r_loss_m = r_loss_final.mean()
        c_loss_m = (-1*c_loss).mean()
        r_c_loss_m = (r_loss_final - c_loss).mean()
        
        return r_c_loss_m, r_loss_m, c_loss_m

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l = UncertainSelectiveMSEAndClass(device)
    outputs = torch.tensor([[0.54, 0.01, 0.99], [0.23, 1.33, 0.12], [0.97, 0.7, 0.3], [0.129, 0.3, 0.7], [0.23, 0.87, 0.13]]).to(device)
    r_l = torch.tensor([[0.53, 0.65, 0.7, 0.1, 0.1]]).T.to(device)
    c_l = torch.tensor([[1, 0, 1, 0, 1]]).T.to(device)
    print(l(outputs, r_l, c_l))