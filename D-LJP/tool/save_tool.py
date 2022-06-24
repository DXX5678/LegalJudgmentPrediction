import torch


def save_checkpoint(output_path, model, optimizer, epoch):
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, output_path)