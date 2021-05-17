"""
This class contain methods to train the
RDetector model
"""
from RDetector import *
from BittiumTrainSetBuilder import *

DEBOG = False


MODEL_NAME = 'Bittium_I'
DATA_NAME = 'Bittium_A'
DEVICE = 'cuda:0'
BATCH_SIZE = 200
LEARNING_RATE = 0.000001
NB_EPOCH = 300
TRAIN_MEMORY = 1000

START_FROM = 7801

if __name__ == '__main__':

    # Build dataset:
    if DEBOG:
        train_memory = 2
    train_set = BittiumTrainSetBuilder(train=True, memory_size=TRAIN_MEMORY, upd_size=1000)
    test_set = BittiumTrainSetBuilder(train=False, memory_size=10, upd_size=3)

    # Get data loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=int(BATCH_SIZE/5),
                                               shuffle=True)
    # The model
    model = RDetector(name=MODEL_NAME).to(DEVICE)
    try:
        model.restore()
        print('Model Successfully restored')
    except:
        print('No model to restore')
    model.train()

    # Loss function according more weights for class 1
    criterion = nn.MSELoss()

    # The optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Get last index if exists
    start_idx = 0
    it = 0
    try:
        df = pd.read_csv('Model/Total_Loss_Track_{}.csv'.format(MODEL_NAME), sep=';', header=None).to_numpy()
        # Start epoch from
        start_idx = int(max(df[:, 0]) + 1)
        # Global index
        it = int(max(df[:, 1]) + 1)

        print('Start from index {} on epoch {}'.format(it, start_idx))
    except:
        print('Start from zero')

    if DEBOG:
        for idx, (signal, target) in enumerate(train_loader):

            # Get a random beat
            rand_index = np.random.randint(low=0, high=signal.shape[0])

            # Predict it
            preds = model(signal.to(DEVICE))
            print(preds)
            print('Preds size:')
            print(preds.size())
            print('input size: ', signal.size())

            # Compare
            x_axis = range(signal.shape[1])
            plt.plot(x_axis, signal[rand_index, :], c='blue')
            plt.plot(x_axis, target[rand_index, :].cpu().detach().numpy()*1000, c='green', linewidth=0.5)
            plt.plot(x_axis, preds[rand_index, :].cpu().detach().numpy()*1000, c='red')

            plt.show()
            plt.close()

            exit(1)

    for i in range(start_idx, start_idx + NB_EPOCH):
        print('=========== Epoch {} / {} =============='.format(i, start_idx + NB_EPOCH))

        epoch_tmp_train_loss = []
        epoch_tmp_test_loss = []
        tmp_train_loss = []
        tmp_test_loss = []
        # Learning step
        model.train()
        for idx, (signal, target) in enumerate(train_loader):

            # Reset grad
            optimizer.zero_grad()
            # Make predictions
            preds = model(signal.to(DEVICE))
            # Reshape target to compare
            target = target.view(target.size()[0] * target.size()[1])
            loss = criterion(preds.view(preds.size(0)*preds.size(1)), target.to(DEVICE))
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()
            it += 1
            tmp_train_loss.append(loss.cpu().detach().item())
            epoch_tmp_train_loss.append(tmp_train_loss[-1])

            # Store each loss in file
            string = '{};{};{}\n'.format(i, it, tmp_train_loss[-1])
            file = open('Model/Total_Loss_Track_{}.csv'.format(MODEL_NAME), 'a')
            file.write(string)
            file.close()


            # Test loss
            if idx % 200 == 0:
                model.eval()
                with torch.no_grad():
                    for idx_test, (signal, target) in enumerate(test_loader):
                        # Make predictions
                        preds = model(signal.to(DEVICE))
                        # Reshape target to compare
                        target = target.view(target.size()[0] * target.size()[1])
                        loss = criterion(preds.view(preds.size(0) * preds.size(1)), target.to(DEVICE))

                    tmp_test_loss.append(loss.cpu().detach().item())
                    epoch_tmp_test_loss.append(tmp_test_loss[-1])
                test_set.update_memory()

                # Get avg loss
                avg_train_loss = np.mean(tmp_train_loss)
                avg_test_loss = np.mean(tmp_test_loss)
                tmp_train_loss = []
                tmp_test_loss = []
                # Store: Epoch/batch_idx/train_loss/test_loss
                string = '{};{};{};{}\n'.format(i, it, avg_train_loss, avg_test_loss)
                file = open('Model/Batch_AVG_Loss_Track_{}.csv'.format(MODEL_NAME), 'a')
                file.write(string)
                file.close()
                print('Epoch {} - Batch {} - AVG train loss: {} - AVG test loss: {}'.format(i, idx,
                                                                                            avg_train_loss,
                                                                                            avg_test_loss))


        epoch_avg_train_loss = np.mean(epoch_tmp_train_loss)
        epoch_avg_test_loss = np.mean(epoch_tmp_test_loss)
        epoch_tmp_train_loss = []
        epoch_tmp_test_loss = []

        print('* ========================================== *')
        print('* Epoch {} - Train Loss {} - Test Loss {}'.format(i, it, epoch_avg_train_loss,
                                                                 epoch_avg_test_loss))
        # Print in the file
        file = open('Model/Epoch_loss_track_{}.csv'.format(MODEL_NAME), 'a')
        file.write('{};{};{}\n'.format(i, epoch_avg_train_loss, epoch_avg_test_loss))
        file.close()
        # Save the model
        model.save()
        # Periodic long term backup
        if i % 1 == 0:
            model.save(epoch=i)

        # Load new elements in the training set
        train_set.update_memory()

