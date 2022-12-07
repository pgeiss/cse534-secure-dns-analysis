import torch
from pipeline import get_train_and_test_for_file
from model import NeuralNetworkClassifier

NUM_LINES = 541

def run_model(model_path,
              validation_path,
              device='cpu'):
    model = NeuralNetworkClassifier()
    model.eval()
    params = torch.load(model_path)
    model.load_state_dict(params)

    _, test = get_train_and_test_for_file([validation_path], NUM_LINES, device=device)

    total_labels = 0
    correct_labels = 0
    for batch, labels in test:
        with torch.no_grad():
            out = model(batch)
            predicted_labels = torch.argmax(out, axis=1)
            for i in range(len(labels)):
                if labels[i] == predicted_labels[i]:
                    correct_labels += 1
            total_labels += len(labels)
    print(f'Tasks:{total_labels}')
    print(f'Correct: {correct_labels}')
    print(f'Accuracy: {correct_labels / total_labels}')
