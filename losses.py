import torch
import torch.nn as nn

class SelectiveMSEAndClass(nn.Module):
    
    """
    Class for loss which adds binary cross entropy loss and MSE loss
    for range predictions. The MSE loss only penalizes incorrect range
    predictions for examples of class 1 (with a call in the spectrogram).
    """
    
    def __init__(self, alpha=1):
        super().__init__()
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCELoss()
        self.alpha = alpha
        
    def forward(self, outputs, r_labels, c_labels):
        
        # Isolate ranges for call-containing example only
        call_outs = outputs[c_labels.squeeze() == 1, 0]
        call_r_labels = r_labels[c_labels.squeeze() == 1]
        
        # Ignore MSE loss entirely if there are no identified calls in the batch for some reason.
        # In practice this shouldn't really happen because randomly initilized weights should
        # produce a 50% classification accuracy as there are an equal number of noise/call examples.
        if len(call_outs) == 0:
            loss = self.BCE(outputs[:,1].squeeze(), c_labels.squeeze())
        else:
            loss = self.alpha*self.MSE(call_outs.squeeze(), call_r_labels.squeeze()) + self.BCE(outputs[:,1].squeeze(), c_labels.squeeze())

        return loss
    
class UncertainSelectiveMSEAndClass(nn.Module):
    
    """
    Class for loss which adds binary cross entropy loss and MSE loss
    for range predictions. The MSE loss only penalizes incorrect range
    predictions for examples of class 1 (with a call in the spectrogram).

    This loss also implements the approach from Kendall et al. (https://arxiv.org/abs/1705.07115) 
    to learn weights for the classification and ranging tasks. 
    """
    
    def __init__(self):
        super().__init__()
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCELoss()

        # Learned variables for weighting the tasks
        self.log_vars = [torch.tensor(0., device=device, requires_grad=True), torch.tensor(0., device=device, requires_grad=True)]
        
    def forward(self, outputs, r_labels, c_labels):
        
        # Isolate ranges for call-containing example only
        call_outs = outputs[c_labels.squeeze() == 1, 0]
        call_r_labels = r_labels[c_labels.squeeze() == 1]
        
        # Get loss
        r_loss = self.MSE(call_outs.squeeze(), call_r_labels.squeeze())
        c_loss = self.BCE(outputs[:,1].squeeze(), c_labels.squeeze())
        
        # Calculate weighted range loss
        r_precision = torch.exp(-self.log_vars[0])
        r_loss = r_precision*r_loss + self.log_vars[0]
        
        # Calculate weighted class loss
        c_precision = torch.exp(-self.log_vars[1])
        c_loss = c_precision*c_loss + self.log_vars[1]
        
        return r_loss + c_loss
