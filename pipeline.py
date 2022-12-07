import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader

ENDS_WITH_CSV = lambda filename: filename.endswith('.csv')
SEED = 0xDEADBEEF

def transform_row(ls):
    row_as_int = [float(st) for st in ls]
    return get_input_and_label_for_row(row_as_int)

def get_input_and_label_for_row(row):
    assert len(row) == 43, f'CSV row did not contain 43 elements: {len(row)}'

    # We have 42 features -- the 43rd item in the CSV is the label
    return (torch.tensor(row[0:42]), torch.tensor(int(row[42])))

def get_train_and_test_for_file(filenames, num_lines, device='cpu'):
    file_opener = dp.iter.FileOpener(filenames, mode='rt')
    parsed_csv = file_opener.parse_csv(delimiter=',')
    mapped_csv = parsed_csv.map(transform_row)

    # In the paper, they use 70% of the data for training and 30% for validation.
    train, test = mapped_csv.random_split(weights={"train": 0.7, "valid": 0.3},
                                          total_length=num_lines,
                                          seed=SEED)
    
    # Shuffle and shard (protection against race conditions)
    shuffled_train = train.shuffle()
    sharded_train = shuffled_train.sharding_filter()
    shuffled_test = test.shuffle()
    sharded_test = shuffled_test.sharding_filter()
    
    train_dataloader = DataLoader(sharded_train, batch_size=16, pin_memory=True, shuffle=True)
    test_dataloader = DataLoader(sharded_test, pin_memory=True, shuffle=True)

    return (train_dataloader, test_dataloader)