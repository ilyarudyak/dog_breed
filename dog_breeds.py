import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

from timeit import default_timer as timer

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_prep import get_loaders


def get_model(n_classes=133):
    use_cuda = torch.cuda.is_available()
    model = models.vgg16(pretrained=True)

    # freeze parameters of the model to avoid brackpropagation
    for param in model.parameters():
        param.requires_grad = False

    # define dog breed classifier part of model_transfer
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(4096, 512),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(512, n_classes))
    model.classifier = classifier

    if use_cuda:
        model = model.cuda()

    return model


def get_model_v2(n_classes=133):
    use_cuda = torch.cuda.is_available()
    model = models.vgg16(pretrained=True)

    # freeze parameters of the model to avoid brackpropagation
    for param in model.parameters():
        param.requires_grad = False

    # define dog breed classifier part of model_transfer
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(4096, 512),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(512, n_classes),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier

    if use_cuda:
        model = model.cuda()

    return model


def get_model_v3(n_classes=133):
    use_cuda = torch.cuda.is_available()
    model = models.vgg16(pretrained=True)

    # freeze parameters of the model to avoid brackpropagation
    for param in model.parameters():
        param.requires_grad = False

    # define dog breed classifier part of model_transfer
    n_inputs = model.classifier[6].in_features
    classifier = nn.Sequential(nn.Linear(n_inputs, 256),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(256, n_classes),
                               nn.LogSoftmax(dim=1))
    model.classifier[6] = classifier

    if use_cuda:
        model = model.cuda()

    return model


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.Inf

    print(f"Batch Size: {loaders['train'].batch_size}\n")

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        # train the model
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch:{epoch}/{n_epochs} \tBatch:{batch_idx + 1}')
                print(f'Train Loss: {train_loss}\n')

        # validate the model
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            loss = criterion(output, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # return trained model
    return model


def train_model(n_epochs=20, batch_size=256):
    loaders = get_loaders(batch_size=batch_size)
    model = get_model()
    use_cuda = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model = train(n_epochs=n_epochs,
                  loaders=loaders,
                  model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  use_cuda=use_cuda,
                  save_path='models/model_transfer.pt')
    return model


def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        # convert output probabilities to predicted class
        output = F.softmax(output, dim=1)
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))


def test_model(model=None, filename='models/model_transfer.pt'):

    if model:
        pass
    else:
        model = get_model()
        model.load_state_dict(torch.load(filename))

    loaders = get_loaders()
    criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    test(loaders=loaders,
         model=model,
         criterion=criterion,
         use_cuda=use_cuda)


def train_v2(model, criterion, optimizer, train_loader, valid_loader,
             save_file_name, save_hist_file, max_epochs_stop=3, n_epochs=20, print_every=1):
    """Train a PyTorch Model
        Params
        --------
            model (PyTorch model): cnn to train
            criterion (PyTorch loss): objective to minimize
            optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
            train_loader (PyTorch dataloader): training dataloader to iterate through
            valid_loader (PyTorch dataloader): validation dataloader used for early stopping
            save_file_name (str ending in '.pt'): file path to save the model state dict
            max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
            n_epochs (int): maximum number of training epochs
            print_every (int): frequency of epochs to print training stats

        Returns
        --------
            model (PyTorch model): trained cnn with best weights
            history (DataFrame): history of train and validation loss and accuracy
    """
    use_cuda = torch.cuda.is_available()
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    epochs_before_stop = 0
    best_epoch = 0
    for epoch in range(n_epochs):
        epochs_before_stop += 1
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu

            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        history.to_csv(save_hist_file, index=False)
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / epochs_before_stop :.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    history.to_csv(save_hist_file, index=False)

    return model, history


def train_model_v2(n_epochs=30, batch_size=256,
                   save_file_name='models/vgg16_v2.pt',
                   save_hist_file='models/vgg16_v2_history.csv'):
    model = get_model_v3()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters())

    loaders = get_loaders(batch_size=batch_size)

    model, history = train_v2(model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              train_loader=loaders['train'],
                              valid_loader=loaders['valid'],
                              save_file_name=save_file_name,
                              save_hist_file=save_hist_file,
                              n_epochs=n_epochs)
    return model, history


if __name__ == '__main__':
    model, history = train_model_v2(n_epochs=30)
    test_model(model)
