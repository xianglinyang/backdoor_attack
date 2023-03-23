import argparse
import os
import time
import datetime
from tqdm import tqdm
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import DataHandler, load_dataset, build_transform, save_dataset, save_sprite
from utils import load_trigger, poison_pair, resize_trigger
from models import BadNet, resnet18


parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, traffic, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--record', '-r', type=float)
parser.add_argument('--device', default='2')
parser.add_argument('--net', "-n", default="BadNet", type=str, choices=["BadNet", "resnet18"])

# poison settings
parser.add_argument('--poison_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_target','-t', type=int)
parser.add_argument('--trigger_source','-s', type=int)
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()


def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, idxs):
    ta = eval(data_loader_val_clean, model, device, idxs)
    asr = eval(data_loader_val_poisoned, model, device, idxs)
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['asr'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, idxs):
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
    acc = (y_true == y_predict).sum()/len(y_true)
    asr = (y_true[idxs] == y_predict[idxs]).sum()/len(y_true[idxs])
    loss = sum(loss_sum) / len(loss_sum)

    return {
            "acc": acc,
            "loss": loss,
            "asr": asr
            }


def main():
    np.random.seed(23478)

    # hyperparameters
    DATASET = args.dataset
    NB_CLASSES = args.nb_classes
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LR = args.lr
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    RECORD_PERIOD = args.record
    NET = args.net

    POISON_RATE = args.poison_rate
    TRIGGER_TARGET = args.trigger_target
    TRIGGER_SOURCE = args.trigger_source
    TRIGGER_SIZE = args.trigger_size

    # path
    save_path = f"/home/xianglin/projects/DVI_data/{NET}_{DATASET}"
    training_path = os.path.join(save_path, "Training_data")
    testing_path = os.path.join(save_path, "Testing_data")
    model_path = os.path.join(save_path, "Model")
    img_path = os.path.join(save_path, "sprites")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(testing_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    print("\n# load patch:")
    trigger = load_trigger()
    trigger = resize_trigger(trigger, TRIGGER_SIZE)

    print("\n# load dataset: %s " % DATASET)
    train_data, train_labels, test_data, test_labels = load_dataset(DATASET)
    poison_X, poison_y, poison_idxs = poison_pair(train_data, train_labels, POISON_RATE, trigger, TRIGGER_SOURCE, TRIGGER_TARGET, np.random.randint(np.iinfo(np.int16).max))
    test_poison_X, test_poison_y, test_poison_idxs = poison_pair(test_data, test_labels, 1.0, trigger, TRIGGER_SOURCE, TRIGGER_TARGET, np.random.randint(np.iinfo(np.int16).max))

    print("Prepare DataHandler:")
    train_transform, test_transform = build_transform(DATASET)
    dataset_poison_train = DataHandler(poison_X, poison_y, train_transform)
    dataset_poison_test = DataHandler(test_poison_X, test_poison_y, test_transform)
    dataset_clean_test = DataHandler(test_data, test_labels, test_transform)
    
    data_loader_train        = DataLoader(dataset_poison_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    data_loader_train_record = DataLoader(dataset_poison_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    data_loader_val_clean    = DataLoader(dataset_clean_test,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    data_loader_val_poisoned = DataLoader(dataset_poison_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) # shuffle 随机化

    # record data

    # # record labels
    with open(os.path.join(save_path, "clean_label.json"), "w") as f:
        json.dump(train_labels.tolist(), f)
    with open(os.path.join(save_path, "noisy_label.json"), "w") as f:
        json.dump(poison_y, f)

    # # record dataset
    save_dataset(data_loader_train_record, training_path, is_train=True)
    save_dataset(data_loader_val_clean, testing_path, is_train=False)

    # # record sprite
    all_data = np.concatenate((poison_X, test_data), axis=0)
    save_sprite(all_data, img_path)

    if NET == "BadNet":
        model = BadNet(input_channels=1 if "MNIST" in DATASET else 3, output_num=NB_CLASSES).to(device)
    elif NET == "resnet18":
        model = resnet18().to(device)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_time = time.time()
    print(f"Start training for {EPOCHS} epochs")

    checkpoint = 1
    idxs = np.arange(len(train_labels)).tolist()
    for epoch in range(EPOCHS):
        running_loss = 0
        model.train()
        period = int(len(data_loader_train)*RECORD_PERIOD)
        batch = 0
        for batch_x, batch_y in tqdm(data_loader_train):
            batch = batch + 1

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(batch_x) # get predict label of batch_x
            
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()
            running_loss += loss

            if batch % period == 0:
                # record
                save_c_path = os.path.join(model_path, f"Checkpoint_{checkpoint}")
                os.makedirs(save_c_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_c_path, "subject_model.pth"))
                with open(os.path.join(model_path, f"Checkpoint_{checkpoint}", "index.json"), "w") as f:
                    json.dump(idxs, f)
                checkpoint = checkpoint + 1

        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, test_poison_idxs)
        scheduler.step()
        print(f"# EPOCH {epoch}   loss: {running_loss.item() / len(data_loader_train):.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()