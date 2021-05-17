import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from BeatClassifier import BeatClassifier
from MIT_Dataset import MIT_Dataset
import Utils
import random

def trainer(debog=False,
            name='BetaClassif_D',
            batch_size=200,
            device='cuda',
            data_device='cpu',
            seed=1,
            nb_epoch=300,
            lr=0.000005,
            periodic_backup=5,      # Model archivate each x epochs
            ):

    # Get the dataset
    train_set = MIT_Dataset(device=data_device,
                            seed=seed,
                            train=True,
                            train_prop=0.8,
                            debog=debog)

    test_set = MIT_Dataset(device=data_device,
                           seed=seed,
                           train=False,
                           copy_from=train_set,
                           debog=debog)

    print('=========== Classes: ==============')
    for key in train_set.classes.keys():
        print('\t key: {} - index: {} - occurences: {}'.format(key,
                                                               train_set.classes[key][1][0],
                                                               len(train_set.classes[key][0])))


    # Data-loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)

    # Model:
    model = BeatClassifier(name=name).to(device)
    # Restore it if present
    try:
        model.restore()
        print('model successfully restored')
    except:
        print('No model to restore')

    # Get weights for crossentropy loss
    weights = np.zeros(len(train_set.classes.keys()))
    to_drop = []
    for key in train_set.classes.keys():
        idx = train_set.classes[key][1][0]  # get the number representation of the key
        weights[idx] = len(train_set.classes[key][0])   # get the number of elements of that label in the dataset
        if weights[idx] < 500:
            to_drop.append(idx)
    tot = np.sum(weights)
    weights = tot / weights
    # Drop small classes:
    for drop in to_drop:
        weights[drop] = 0
    weights = torch.Tensor(weights).to(device)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    # Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Load optimizer state if exists
    try:
        optimizer.load_state_dict(torch.load('Model/Weights_{}_optimizer.pt'.format(model.name)))
        print('Optimizer state successfully restored')
    except:
        print('No optimizer state to restore')

    start_epoch = 0
    # Check if history present:
    try:
        history = pd.read_csv('Model/{}-LearningTrack.csv'.format(name), header=None, sep=';').to_numpy()
        start_epoch = np.max(history[:, 0]) + 1
        print('Start from index {}'.format(start_epoch))
    except:
        print('No model to load. Starting from zero')

    end_epoch = start_epoch + nb_epoch

    if debog:
        loop_idx = 0
        for idx, (beats, targets, index) in enumerate(train_loader):

            rdn_idx = random.randint(0, targets.size(0))
            x = range(beats.size(1))
            plt.plot(x, beats[rdn_idx, :].cpu().detach().numpy())
            plt.show()
            if loop_idx >= 5:
                exit(-1)
            loop_idx += 1


    save_loss = []
    for e in range(int(start_epoch), int(end_epoch)):
        roc = False
        if (e % 5 == 0 and e != 0) or e == end_epoch - 1:
            roc = True
        print('Epoch {} / {}'.format(e, end_epoch))

        # Training part
        tmp_train_loss = []
        tmp_train_acc = []
        model.train()
        for idx, (beats, targets, index) in enumerate(train_loader):
            #torch.cuda.empty_cache()
            optimizer.zero_grad()
            # Get data
            beats = beats.to(device)
            targets = targets.to(device)

            # Make predictions
            preds = model(beats)
            # Get the loss
            loss = loss_fn(preds, targets)
            # Optimization step
            loss.backward()
            optimizer.step()

            # Tracking
            class_pred = torch.argmax(preds, dim=1)
            acc = (class_pred == targets).sum().cpu().detach().item() / targets.size(0)
            tmp_train_acc.append(acc)
            tmp_train_loss.append(loss.cpu().detach().item())

            string = '{};{};{}\n'.format(e, idx, loss.cpu().detach().item())
            file = open('model/{}_total_loss_track.csv'.format(name), 'a')
            file.write(string)
            file.close()

            if idx % 10 == 0 and idx != 0:
                print('\t batch {} - loss {} '.format(idx, np.mean(tmp_train_loss[-10:-1])))
                save_loss.append(tmp_train_loss[-1])


        avg_tmp_train_loss = np.mean(tmp_train_loss)
        avg_tmp_train_acc = np.mean(tmp_train_acc)
        print('Train Loss: {} - Train Accuracy: {}'.format(avg_tmp_train_loss,
                                                           avg_tmp_train_acc))

        # Test part
        tmp_test_loss = []
        tmp_test_acc = []

        # Store elements for the roc curve
        if roc:
            roc_prds = np.zeros((len(test_set), model.nb_classes))
            roc_targets = np.zeros((len(test_set)))
            roc_classes = test_set.classes
            roc_idx = 0
        for idx, (beats, targets, index) in enumerate(test_loader):
            # Get data
            beats = beats.to(device)
            targets = targets.to(device)
            # Make predictions
            model.eval()
            with torch.no_grad():
                preds = model(beats)
                loss = loss_fn(preds, targets)
                class_pred = torch.argmax(preds, dim=1)
                acc = (class_pred == targets).sum().cpu().detach().item() / targets.size(0)
                tmp_test_acc.append(acc)
                tmp_test_loss.append(loss.cpu().detach().item())

                # Store for roc curves
                if roc:
                    nb = preds.size(0)
                    roc_prds[roc_idx:roc_idx+nb, :] = torch.softmax(preds, dim=1).cpu().detach().numpy()
                    roc_targets[roc_idx:roc_idx+nb] = targets.cpu().detach().numpy()
                    roc_idx = roc_idx + nb
        avg_tmp_test_loss = np.mean(tmp_test_loss)
        avg_tmp_test_acc = np.mean(tmp_test_acc)
        print('Test Loss: {} - Test Accuracy: {}'.format(avg_tmp_test_loss,
                                                         avg_tmp_test_acc))

        # Write in a file
        string = [str(e),
                  str(avg_tmp_train_loss),
                  str(avg_tmp_train_acc),
                  str(avg_tmp_test_loss),
                  str(avg_tmp_test_acc)]
        file = open('Model/{}-LearningTrack.csv'.format(name), 'a')
        file.write('{}\n'.format(';'.join(string)))
        file.close()

        # Plot ROC curves
        if roc:
            Utils.RocCurves(roc_targets, roc_prds, roc_classes, name, e)

        # Save the model
        model.save(epoch=e)
        # Save optimizer state
        torch.save(optimizer.state_dict(), 'Model/Weights_{}_optimizer.pt'.format(model.name))
        torch.save(optimizer.state_dict(), 'Model/Weights_{}_epoch_{}_optimizer.pt'.format(model.name, e))

    plt.plot([i for i in range(len(save_loss))], save_loss)
    plt.show()

if __name__ == '__main__':

    trainer()















