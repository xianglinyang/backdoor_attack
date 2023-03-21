import argparse
import os
import re
import time
import datetime

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from dataset import DataHandler, load_dataset, load_trigger, build_transform, poison_pair, resize_trigger

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, traffic, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--device', default='1')

# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_target','-t' type=int)
parser.add_argument('--trigger_source','-s' type=int)
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()

def train_one_epoch(data_loader, model, criterion, optimizer, loss_mode, device):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return {
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device):
    ta = eval(data_loader_val_clean, model, device, print_perform=True)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }


def main():
    np.random.seed(23478)

    # hyperparameters
    DATASET = args.dataset
    NB_CLASSES = args.nb_classes
    LOSS = args.loss
    OPTIMIZER = args.optimizer
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LR = args.lr
    DATA_PATH = args.data_path
    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    POISON_RATE = args.poison_rate
    TRIGGER_TARGET = args.trigger_target
    TRIGGER_SOURCE = args.trigger_source
    TRIGGER_SIZE = args.trigger_size

    print("\n# load patch:")
    trigger = load_trigger()
    trigger = resize_trigger(trigger, TRIGGER_SIZE)

    print("\n# load dataset: %s " % DATASET)
    clean_trainset, clean_testset = load_dataset(DATASET)
    poison_X, poison_y, poison_idxs = poison_pair(clean_trainset.data, clean_trainset.targets, POISON_RATE, trigger, TRIGGER_SOURCE, TRIGGER_TARGET, np.random.randint(np.iinfo(np.int16).max))
    test_poison_X, test_poison_y, test_poison_idxs = poison_pair(clean_testset.data, clean_testset.targets, 1.0, trigger, TRIGGER_SOURCE, TRIGGER_TARGET, np.random.randint(np.iinfo(np.int16).max))

    print("Prepare DataHandler:")
    train_transform, test_transform = build_transform(DATASET)
    dataset_poison_train = DataHandler(poison_X, poison_y, train_transform)
    dataset_poison_test = DataHandler(test_poison_X, test_poison_y, test_transform)
    dataset_clean_test = DataHandler(clean_testset.data, clean_testset.targets, test_transform)
    
    data_loader_train        = DataLoader(dataset_poison_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_clean_test,   batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_poison_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers) # shuffle 随机化

    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    start_time = time.time()
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
        # save model 
        torch.save(model.state_dict(), basic_model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()