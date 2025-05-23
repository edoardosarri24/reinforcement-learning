import torch
import os

def save_checkpoint(episode, model, dir):
    '''
        Saves the model state to a file named with the episode number.
        Args:
            episode (int): The current episode number.
            model (torch.nn.Module): The model to save the state of.
            dir (str): The directory where the checkpoint will be saved.
    '''
    torch.save(
        {'episode': episode,
         'model_state_dict': model.state_dict()},
         os.path.join(dir, f'checkpoint-{episode}.pt'))

def restore_checkpoint(dir, episode, model, device):
    '''
        Restores the model state from a checkpoint file.
        Args:
            checkpoint_path (str): The path to the checkpoint file.
            episode (int): The episode of the best model.
            model (torch.nn.Module): The model to restore the state of.
            device (torch.device): The device where the model will be loaded.
        Returns:
            model (torch.nn.Module): The restored model.
    '''
    checkpoint_path = os.path.join(dir, f'checkpoint-{episode}.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def split_vector(vec:list[int], size:int=2) -> list[list[int]]:
    result = [vec[i:i+size] for i in range(0, len(vec)-1, size-1)]
    if len(result[-1]) == 1:
        result[-2].append(result.pop()[0])
    return result