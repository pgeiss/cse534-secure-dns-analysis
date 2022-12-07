import torch
import copy
from pipeline import get_train_and_test_for_file
from model import NeuralNetworkClassifier
from enum import Enum

NUM_LINES = 541

class OptimType(Enum):
    DEFAULT=1
    ADAGRAD=2
    ADAM=3

def train_model(file_path,
                optim_type=OptimType.ADAM,
                num_classes=11,
                num_epochs=10,
                device='cpu'):
    last_forward_slash_idx = file_path.rfind('/')
    last_period_idx = file_path.rfind('.')
    filename = file_path[last_forward_slash_idx+1:last_period_idx]
    train, test = get_train_and_test_for_file([file_path], NUM_LINES, device=device)

    model = NeuralNetworkClassifier(num_classes=num_classes, device=device)
    model_path = f'models/{filename}'
    best_validation_result = torch.inf
    print(f'Beginning training for {num_epochs} epochs...')
    match optim_type:
        case OptimType.ADAM:
            optim = torch.optim.Adam(model.parameters(), lr=0.0000001)
        case OptimType.ADAGRAD:
            optim = torch.optim.Adagrad(model.parameters())
        case _:
            optim = torch.optim.SGD(model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')

        #### Training Phase
        model.train()
        training_loss = float(0)
        for idx, (batch, labels) in enumerate(train):
            batch.to(device)
            labels.to(device)
            logits = model(batch)
            
            cur_loss = model.loss(logits, labels)
            optim.zero_grad()
            cur_loss.backward()
            optim.step()
            
            training_loss += cur_loss.item()
            if idx % 100 == 0:
                size = float(idx + 1)
                # print(f'Average loss: {training_loss / size}')
        size = float(idx + 1)
        avg_train_loss = training_loss / size
        print(f'Average loss: {avg_train_loss}')

        #### Validation Phase
        model.eval()
        validation_loss = float(0)
        for idx, (batch, labels) in enumerate(test):
            logits = model(batch)
            cur_loss = model.loss(logits, labels)
            validation_loss += cur_loss.item()
        size = float(idx + 1)
        avg_validation_loss = validation_loss / size
        print(f'Average validation loss: {avg_validation_loss}')
        if avg_validation_loss < best_validation_result:
            best_validation_result = avg_validation_loss
            epoch_model_path = f'{model_path}_{epoch}.pt'
            print(f'Saving Epoch #{epoch} result to {epoch_model_path}')
            torch.save(copy.deepcopy(model.state_dict()), epoch_model_path)
            print('Saved.')