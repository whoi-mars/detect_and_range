import torch
from tqdm import tqdm
import time
import copy
import os
import math
import dataLoader
import losses
from tcn import BranchedTCN, TCN
import config
import argparse
import wandb

parser = argparse.ArgumentParser(description="TCN Training on KRAKEN Synthetic Data")
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (default: 0.2)')
parser.add_argument('--start_epoch', type=int, default=1,
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
parser.add_argument('--lr_decay', action='store_true',
                    help='decays learning rate via lr /= 10 when epoch % 10 = 0')
parser.add_argument('--verbose', action='store_true',
                    help='display progress while training (default: False)')
parser.add_argument('--id', type=str,
                    help='id of run to continue'),
parser.add_argument('--checkpoint_dir', type=str, default='TCN',
                    help='directory where model weight checkpoints will be saved (default: TCN)')
parser.add_argument('--source_dir', type=str,
                    help='optional directory to source model from to continue training and save to another directory')
parser.add_argument('--resume', action='store_true',
                    help='resume previous weights and biases run (default: False)')
parser.add_argument('--no_wandb', action='store_false',
                    help='disables weights and biases logging (default: True')
parser.add_argument('--data_parallel', action='store_true',
                    help='wraps the model in a torch.nn.DataParallel object to enable training with multiple GPUs (default: False)')
parser.add_argument('--freeze_class', action='store_true',
                    help='freezes the parameters associated with the class prediction (default: False)')
parser.add_argument('--uncertain_loss', action='store_true',
                    help='uses uncertainty loss (default: False)')
parser.add_argument('--hpf', action='store_true',
                    help='uses high pass filter on data')
args = parser.parse_args()

# log in to wandb and initialize
if args.no_wandb:
    wandb.login()
    wandb.init(
        project="tcn-test",
        config={
            "epochs": args.end_epoch - args.start_epoch + 1,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "ksize": args.ksize,
            "levels": args.levels,
            "nhid": args.nhid,
            "alpha": args.alpha,
            "nperseg": config.nperseg,
            "noverlap": config.noverlap,
            "nfft": config.nfft,
            "model_dir": args.checkpoint_dir
        },
        id=args.id,
        resume=args.resume)

# Set seed
torch.manual_seed(args.seed)

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                                squeeze=True,
                                hpf=args.hpf)
n_steps_per_epoch = math.ceil(len(dl['train'].dataset) / args.batch_size)

# Create TCN model
channel_sizes = [args.nhid] * args.levels
n_outputs = 2
input_channels = dl['train'].dataset.imsize[0]

# Print size of input
print(f"Spectrogram size: {dl['train'].dataset.imsize}")

# Initialize model
model = BranchedTCN(input_size=input_channels, output_size=n_outputs, num_channels=channel_sizes, kernel_size=args.ksize, dropout=args.dropout).to(device)
#model = TCN(input_size=input_channels, output_size=n_outputs, num_channels=channel_sizes, kernel_size=args.ksize, dropout=args.dropout).to(device)

# Freeze class parameters
if isinstance(model, BranchedTCN) and args.freeze_class:
    model.freeze_class()
    print("Class Prediction Weights Frozen")

# Save directory for modle weights
save_dir = os.path.join(config.models_dir, args.checkpoint_dir)
os.makedirs(save_dir, exist_ok=True)

# Create loss and optimizer
# UNCERTAIN
# criterion = losses.UncertainSelectiveMSEAndClass()
# parameters = ([p for p in model.parameters()] + [criterion.log_vars[0]] + [criterion.log_vars[1]])
# optimizer = torch.optim.Adam(parameters, lr=args.lr)
# CERTAIN
# criterion = losses.SelectiveMSEAndClass(alpha=args.alpha)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Load weights to start training at start epoch
if args.start_epoch > 1:

    # Check if alternate source directory
    if args.source_dir is not None:
        source_dir = os.path.join(config.models_dir, args.source_dir)
        print(f"Sourceing Weights From: {source_dir}")
    else:
        source_dir = save_dir

    try:
        # Load checkpoint dict
        checkpoint = torch.load(os.path.join(source_dir, f'weights_{args.start_epoch - 1}.pt'))
        # Set up uncertainty loss or normal loss from checkpoint
        if args.uncertain_loss:
            # Set up loss function loading learned log_vars
            criterion = losses.UncertainSelectiveMSEAndClass(checkpoint['log_vars'])
            # Set up optimizer
            parameters = ([p for p in model.parameters()] + [criterion.log_vars[0]] + [criterion.log_vars[1]])
        else:
            criterion = losses.SelectiveMSEAndClass(alpha=args.alpha)
            parameters = model.parameters()
        # Create optimizer
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        # Load optimizer and model states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        raise ValueError("Desired start epoch does not have a corresponding set of saved model weights.")
else:
    # Set up uncertainty loss or normal loss from scratch
    if args.uncertain_loss:
        criterion = losses.UncertainSelectiveMSEAndClass()
        parameters = ([p for p in model.parameters()] + [criterion.log_vars[0]] + [criterion.log_vars[1]])
    else:
        criterion = losses.SelectiveMSEAndClass(alpha=args.alpha)
        parameters = model.parameters()
    # Set up optimizer
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

if args.data_parallel:
    model = torch.nn.DataParallel(model)

def get_model_state_dict(model):

    """
    Save the model state dictionary, taking into account whether
    or not the model is wrapped in a torch.nn.DataParallel object.

    Parameters
    ----------
    model: torch.nn.Module, model to save state dict for.

    Returns
    -------
    state_dict, state dictionary for model.
    """

    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()

def train(model, dataloaders, criterion, optimizer, num_epochs, max_range, save_dir=None, save_all_epochs=False, start_epoch=1):
    
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
    best_model_wts = copy.deepcopy(get_model_state_dict(model))
    best_opt_state = copy.deepcopy(optimizer.state_dict())
    best_log_vars = criterion.log_vars
    best_mse = float('inf')
    
    for epoch in range(start_epoch, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-'*10)

        # reduce learning rate every 10th epoch
        if epoch % 10 == 0 and args.lr_decay:
            args.lr /= 10
            print("decreasing learning rate to: ", args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
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
            running_r_loss = 0.0
            running_c_loss = 0.0
            running_sq_error = 0.0
            running_corrects = 0
            running_call_count = 0
            
            # Flip through data batches
            for step, (inputs, r_labels, c_labels) in enumerate(tqdm(dataloaders[phase], disable=not args.verbose)):
                
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
                    loss, r_loss, c_loss = criterion(outputs, r_labels, c_labels)
                    
                    # Backprop if we are training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Update running statistics -- sq_error only accumulated for examples with calls
                running_loss +=  loss.item() * inputs.size(0)
                running_c_loss += c_loss.item() * inputs.size(0)
                running_r_loss += r_loss.item() * inputs.size(0)
                running_sq_error += ((max_range*r_labels[torch.where((outputs[:,1].squeeze() >= 0.5) & (c_labels.squeeze() == 1))].squeeze() - max_range*outputs[torch.where((outputs[:,1].squeeze() >= 0.5) & (c_labels.squeeze() == 1))[0],0].squeeze()) ** 2).sum().item()      
                running_corrects += torch.sum((outputs[:,1].squeeze() >= 0.5) == c_labels.squeeze())
                running_call_count += torch.sum(c_labels.squeeze())

                # Step-dependent wandb updates during training
                if args.no_wandb:
                    if phase == 'train':
                        step_metrics = {"train/train_loss": loss,
                                        "train/range_loss": r_loss,
                                        "train/class_loss": c_loss,
                                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                                        "train/log_var_0": criterion.log_vars[0],
                                        "train/log_var_1": criterion.log_vars[1]}
                        if step + 1 < n_steps_per_epoch:
                            wandb.log(step_metrics)

            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_c_loss = running_c_loss / len(dataloaders[phase].dataset)
            epoch_r_loss = running_r_loss / len(dataloaders[phase].dataset)
            epoch_mse = running_sq_error / running_call_count
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            
            # wandb logging for end of epoch
            if args.no_wandb:
                if phase == 'train':
                    train_metrics = {"train/train_avg_loss": epoch_loss,
                                    "train/train_avg_range_loss": epoch_r_loss,
                                    "train/train_avg_class_loss": epoch_c_loss,
                                    "train/train_rmse": torch.sqrt(epoch_mse) / 1000,
                                    "train/train_accuracy": epoch_acc}
                    wandb.log({**step_metrics, **train_metrics})
                else:
                    val_metrics = {"val/val_avg_loss": epoch_loss,
                                "val/val_avg_range_loss": epoch_r_loss,
                                "val/val_avg_class_loss": epoch_c_loss,
                                "val/val_rmse": torch.sqrt(epoch_mse) / 1000,
                                "val/val_accuracy": epoch_acc}
                    wandb.log(val_metrics)

            # Print epoch info
            print("{} Loss: {:.4f} -- MSE: {:.4f} km^2 -- RMSE: {:.4f} km -- ACC: {:.4f} -- LV1: {:.4f} -- LV2: {:.4f}".format(phase, epoch_loss, epoch_mse / 1000000, torch.sqrt(epoch_mse) / 1000, epoch_acc, criterion.log_vars[0], criterion.log_vars[1]))

            # Deep copy model
            if phase == 'val' and epoch_mse < best_mse:
                best_mse = epoch_mse
                best_model_wts = copy.deepcopy(get_model_state_dict(model))
                best_opt_state = copy.deepcopy(optimizer.state_dict())
                best_log_vars = criterion.log_vars
            if phase == 'train':
                train_mse_history.append(epoch_mse / 1000000)
                train_rmse_history.append(torch.sqrt(epoch_mse) / 1000)
                train_acc_history.append(epoch_acc)
            if phase == 'val':
                val_mse_history.append(epoch_mse / 1000000)
                val_rmse_history.append(torch.sqrt(epoch_mse) / 1000)
                val_acc_history.append(epoch_acc)
            if phase == 'train' and save_all_epochs:
                torch.save({'model_state_dict': get_model_state_dict(model), 
                            'optimizer_state_dict': optimizer.state_dict(),
                            'log_vars': criterion.log_vars,
                           }, os.path.join(save_dir, 'weights_{}.pt'.format(epoch)))

        print()
        
    # Training done!
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
    print('Best val MSE: {:.4f} km^2'.format(best_mse / 1000000))
    print('Best val RMSE: {:.4f} km'.format(torch.sqrt(best_mse) / 1000))
    
    # Save best model weights and load them into the model before returning
    torch.save({'model_state_dict': best_model_wts,
                'optimizer_state_dict': best_opt_state,
                'log_vars': criterion.log_vars,
                }, os.path.join(save_dir, 'weights_best_val_mse.pt'))
    if not save_all_epochs:
        torch.save({'model_state_dict': get_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'log_vars': criterion.log_vars, 
                   }, os.path.join(save_dir, 'weights_last_{}.pt'.format(epoch)))
    
    history = {'train_mse' : train_mse_history, 
               'train_rmse' : train_rmse_history, 
               'val_mse' : val_mse_history, 
               'val_rmse' : val_rmse_history,
               'train_acc' : train_acc_history,
               'val_acc' : val_acc_history}
    
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