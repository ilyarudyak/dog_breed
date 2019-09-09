import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_prep import get_loaders


def get_model(n_classes=133):
    use_cuda = torch.cuda.is_available()
    model_transfer = models.vgg16(pretrained=True)

    # freeze parameters of the model to avoid brackpropagation
    for param in model_transfer.parameters():
        param.requires_grad = False

    # define dog breed classifier part of model_transfer
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(4096, 512),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(512, n_classes))
    model_transfer.classifier = classifier

    if use_cuda:
        model_transfer = model_transfer.cuda()

    return model_transfer


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


def train_model(n_epochs=10):
    loaders = get_loaders(batch_size=256)
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


if __name__ == '__main__':
    train_model()
