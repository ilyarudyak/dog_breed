import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.autograd import Variable

from data_prep import get_files

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_vgg():
    # define VGG16 model
    VGG16 = models.vgg16(pretrained=True)

    # check if CUDA is available
    use_cuda = torch.cuda.is_available()

    # move model to GPU if CUDA is available
    if use_cuda:
        VGG16 = VGG16.cuda()

    return VGG16


def VGG16_predict(img_path):
    """
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    """
    VGG16 = get_vgg()

    img = Image.open(img_path)
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    img = data_transform(img)
    img = img.unsqueeze(0)
    img = Variable(img)

    prediction = VGG16(img)
    prediction = prediction.data.numpy().argmax()

    return prediction


def is_dog(img_path):
    class_index = VGG16_predict(img_path)
    if 151 <= class_index <= 268:
        return True
    else:
        return False


def get_accuracy(files, n=100):
    return sum([is_dog(file) for file in files]) / n


if __name__ == '__main__':
    human_files, dog_files = get_files()
    human_files_short = human_files[:100]
    dog_files_short = dog_files[:100]
    print(1 - get_accuracy(human_files_short))
    print(get_accuracy(dog_files_short))


