import os
import argparse
from data_loader import create_loaders
from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, help='input batch size', default=64)
parser.add_argument('--epochs', type=int, help='number of epochs for training', default=80)
parser.add_argument('--nz', type=int, help='length of a noise vector', default=100)
parser.add_argument('--alpha', type=float, help='negative slope for leaky ReLU', default=.2)
parser.add_argument('--drop_rate', type=float, help='rate for dropout layers', default=.5)
parser.add_argument('--ngf', type=int,
                    help='generator multiplier for convolution transpose output layers', default=64)
parser.add_argument('--ndf', type=int,
                    help='multiplier for convolution output layers', default=64)
parser.add_argument('--learning_rate', type=float, help='learning rate', default=.0002)
parser.add_argument('--beta1', type=float, help='beta1 for adam optimizer', default=.5)
parser.add_argument('--out_dir', type=str,
                    help='output directory to hold the saved model and generated images',
                    default='./train_out')
parser.add_argument('--dataset_dir', type=str,
                    help='output directory to hold the downloaded SVHN dataset',
                    default='./svhn')

opt = parser.parse_args()
print(opt)

image_size = 32


def main():
    svhn_loader_train, svhn_loader_test = create_loaders(image_size, opt.batch_size,
                                                         opt.dataset_dir, opt.num_workers)

    solver = Solver(svhn_loader_train, svhn_loader_test, opt)
    if os.path.exists(solver.best_netG_filename) and os.path.exists(solver.best_netD_filename):
        solver.load_model(solver.best_netG_filename, solver.best_netD_filename)
        solver.test(1, 1)
    else:
        solver.train()

if __name__ == "__main__":
    main()
