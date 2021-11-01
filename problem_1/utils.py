import torch.optim as optim
import torch

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def load_optimizer(model, config):
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    if config['name'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif config['name'] == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=momentum)
    return optimizer
