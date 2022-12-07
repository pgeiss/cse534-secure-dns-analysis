import sys
import numpy as np
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

if not (3 < len(sys.argv) < 5):
    print('Usage: python3 trees.py <filename> <tree|extra|forest|adaboost> [<transfer-learning filename>]')
    sys.exit(1)

filename = sys.argv[1]
operation = sys.argv[2]
if len(sys.argv) > 3:
    transfer_filename = sys.argv[3]
else:
    transfer_filename = None

data = []
labels = []
with open(filename, 'r') as f:
    for line in f:
        line_as_num = [float(st) for st in line.split(',')]
        assert len(line_as_num) == 43, f'CSV row did not contain 43 elements: {len(line_as_num)}'
        data.append(line_as_num[0:42])
        labels.append(line_as_num[42])

data = np.array(data)
labels = np.array(labels)
assert data.shape[0] == labels.shape[0], f'{data.shape[0]}, {labels.shape}'

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=0xDEADBEEF)

match operation:
    case 'tree':
        clf = tree.DecisionTreeClassifier()
    case 'extra':
        clf = ExtraTreesClassifier()
    case 'forest':
        clf = RandomForestClassifier()
    case 'adaboost':
        clf = AdaBoostClassifier()

clf = clf.fit(train_data, train_labels)
predictions = clf.predict(test_data)
correct_predictions = sum(predictions[i] == test_labels[i] for i in range(len(predictions)))
f1 = f1_score(test_labels, predictions, average='macro')
print(f'Tasks:{len(test_labels)}')
print(f'Correct: {correct_predictions}')
print(f'F1 Score: {f1}')

if transfer_filename is None:
    sys.exit(0)

print('======= Transfer Learning =======')
print(f'{transfer_filename} applied to {filename}')

data = []
labels = []
with open(transfer_filename, 'r') as f:
    for line in f:
        line_as_num = [float(st) for st in line.split(',')]
        assert len(line_as_num) == 43, f'CSV row did not contain 43 elements: {len(line_as_num)}'
        data.append(line_as_num[0:42])
        labels.append(line_as_num[42])

data = np.array(data)
labels = np.array(labels)
assert data.shape[0] == labels.shape[0], f'{data.shape[0]}, {labels.shape}'
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=0xDEADBEEF)
predictions = clf.predict(test_data)
correct_predictions = sum(predictions[i] == test_labels[i] for i in range(len(predictions)))
f1 = f1_score(test_labels, predictions, average='macro')
print(f'Tasks:{len(test_labels)}')
print(f'Correct: {correct_predictions}')
print(f'F1 Score: {f1}')