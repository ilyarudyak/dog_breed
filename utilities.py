import matplotlib.pyplot as plt
import pandas as pd


def plot_losses(history):
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(
            history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')


def plot_acc(history):
    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(
            100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')


if __name__ == '__main__':
    history = pd.read_csv('models/vgg16_transfer_v2_history.csv')
    plot_losses(history)
    plt.show()
