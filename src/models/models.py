from src.models.cifar_resnet import CifarResNet18
from src.models.mnist_cnn import Mnist_CNN


def choose_model(options):
    model_name = str(options['model_name']).lower()
    if model_name == 'mnist_cnn':
        return Mnist_CNN()
    if model_name == 'cifar_resnet18':
        return CifarResNet18()
    raise ValueError('Unsupported model: {}'.format(options['model_name']))

