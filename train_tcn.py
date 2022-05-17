import torch
from tqdm import tqdm
import time
import copy
import dataLoader
import losses
from tcn import TCN
import config
import argparse

parser = argparse.ArgumentParser(description="TCN Training on KRAKEN Synthetic Data")
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 64')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (default: 0.2)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch number to start at (default: 0)')
parser.add_argument('--end_epoch', type=int, default=10,
                    help='epoch number to end at (default: 10)')
parser.add_argument('--ksize', type=int, default=4, metavar='KERNEL SIZE',
                    help='kernel size (default: 4)')
parser.add_argument('--levels', type=int, default=10,
                    help='# of levels (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--nhid', type=int, default=368,
                    help='number of hidden units per layer (default: 368)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--save_all_epochs', action='store_true',
                    help='store weights for all epochs during training (default: False)')
parser.add_argument('--alpha', type=float, default=2, metavar='RANGE LOSS WEIGHT',
                    help='weight for range loss (default: 2)')
parser.add_argument('--verbose', action='store_true',
                    help='display progress while training (default: False)')
args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)

# Get device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")

# Get dataloaders
dl = dataLoader.get_dataloaders(data_dir=config.data_dir,
                                batch_size=args.batch_size,
                                max_range=config.max_range,
                                shuffle=True,
                                transform=dataLoader.get_image_transforms(),
                                squeeze=True)

# Create TCN model
channel_sizes = [args.nhid] * args.levels
n_outputs = 2
input_channels = 232
model = TCN(input_size=input_channels, output_size=n_outputs, num_channels=channel_sizes, kernel_size=args.ksize, dropout=args.dropout).to(device)

# Save directory for modle weights
save_dir = config.models_dir + '/TCN'

# Load weights to start training at start epoch
if args.start_epoch:
    try:
        model.load_state_dict(torch.load(os.path.join(save_dir, f'weights_{args.start_epoch}.pt')))
    except:
        ValueError("Desired start epoch is not have a corresponding set of saved model weights.")

# Create loss and optimizer
criterion = losses.SelectiveMSEAndClass(alpha=args.alpha)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(model, dataloaders, criterion, optimizer, num_epochs, max_range, save_dir=None, save_all_epochs=False, start_epoch=0):
    
    """
    Helper function with some features to train the model and save the most
    recent epoch with the lowest validation MSE.
    
    Parameters
    ----------
    model: nn.Module, model object.
    dataloaders: dict, dictionary containing at least the keys
                 'train', 'val', and 'test' which map to dataloaders
                 for these datasets.
    criterion: torch.nn.Loss, Loss function.
    optimizer: torch.optim, method of weight updating.
    num_epochs: int, number of epochs to train the model.
    max_range: float, maximum simulated range.
    save_dir: str, directory where models will be saved to. Make
              None to not write anything to disk.
    save_all_epochs: bool, whether to save the model weights for all
                     epochs, or just the best MSE weights.
    
    Returns
    -------
    model: nn.Module, model object with weights for best validation set MSE.
    val_mse_history: list, validation set MSE over all epochs
    train_mse_history: list, train set MSE over all epochs
    """
    
    # Time training
    since = time.time()
    
    # History vectors
    val_mse_history = []
    val_rmse_history = []
    val_acc_history = []
    train_mse_history = []
    train_rmse_history = []
    train_acc_history = []
    
    
    # Initialize best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mse = float('inf')
    
    for epoch in range(start_epoch - 1, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-'*10)
        
        # Train + evaluate on validation data each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                # Put model in train mode
                model.train()
            else:
                # Put model in evaluation mode
                model.eval()
            
            # Keep track of loss, raw error, and number of calls
            running_loss = 0.0
            running_sq_error = 0.0
            running_corrects = 0
            running_call_count = 0
            
            # Flip through data batches
            for inputs, r_labels, c_labels in tqdm(dataloaders[phase]):
                
                # Put data/labels on device
                inputs = inputs.to(device)
                r_labels = r_labels.to(device)
                c_labels = c_labels.to(device)
                
                # Zero out gradient for new batch
                optimizer.zero_grad()
                
                # Update weights
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # Get model outputs and loss
                    outputs = model(inputs)
                    loss = criterion(outputs, r_labels, c_labels)
                    
                    # Backprop if we are training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Update running statistics -- sq_error only accumulated for examples with calls
                running_loss +=  loss.item() * inputs.size(0)
                running_sq_error += ((max_range*r_labels[torch.where((outputs[:,1].squeeze() >= 0.5) & (c_labels.squeeze() == 1))].squeeze() - max_range*outputs[torch.where((outputs[:,1].squeeze() >= 0.5) & (c_labels.squeeze() == 1))[0],0].squeeze()) ** 2).sum().item()      
                running_corrects += torch.sum((outputs[:,1].squeeze() >= 0.5) == c_labels.squeeze())
                running_call_count += torch.sum(c_labels.squeeze())
                
            # Stats to display
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_mse = running_sq_error / running_call_count
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            
            # Print epoch info
            print("{} Loss: {:.4f} -- MSE: {:.4f} km^2 -- RMSE: {:.4f} km -- ACC: {:.4f}".format(phase, epoch_loss, epoch_mse / 1000000, torch.sqrt(epoch_mse) / 1000, epoch_acc))

            # Deep copy model
            if phase == 'val' and epoch_mse < best_mse:
                best_mse = epoch_mse
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_mse_history.append(epoch_mse / 1000000)
                train_rmse_history.append(torch.sqrt(epoch_mse) / 1000)
                train_acc_history.append(epoch_acc)
            if phase == 'val':
                val_mse_history.append(epoch_mse / 1000000)
                val_rmse_history.append(torch.sqrt(epoch_mse) / 1000)
                val_acc_history.append(epoch_acc)
            if phase == 'train' and save_all_epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, 'weights_{}.pt'.format(epoch + 1)))
            
        print()
        
    # Training done!
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val MSE: {:.4f} km^2'.format(best_mse / 1000000))
    print('Best val RMSE: {:.4f} km'.format(torch.sqrt(best_mse) / 1000))
    
    # Save best model weights and load them into the model before returning
    torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_mse.pt'))
    if not save_all_epochs:
        torch.save(model.state_dict(), os.path.join(save_dir, 'weights_last.pt'.format(epoch + 1)))
    model.load_state_dict(best_model_wts)
    
    history = {'train_mse' : train_mse_history, 'train_rmse' : train_rmse_history, 'val_mse' : val_mse_history, 'val_rmse' : val_rmse_history}
    
    return model, history

if __name__ == "__main__":
    # Train model
    model, history = train(model=model,
                           dataloaders=dl,
                           criterion=criterion,
                           optimizer=optimizer,
                           start_epoch=args.start_epoch,
                           num_epochs=args.end_epoch,
                           max_range=config.max_range,
                           save_dir=save_dir,
                           save_all_epochs=args.save_all_epochs)