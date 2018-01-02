import data_loader
from solver import Solver

def main():
    print('In main')
    image_size = 32
    batch_size = 64

    svhn_loader_train, svhn_loader_test = data_loader.get_loader(image_size, batch_size)
    solver = Solver(svhn_loader_train, svhn_loader_test, batch_size)

    solver.train()


main()
print('Finished')
