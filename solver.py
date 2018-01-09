'''
Solver composes the generator and discriminator network, defines optimizers and
losses and contains method for train and test
'''
import os
import shutil
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
from model import _netG, _netD


class Solver:
    '''
    The solver does the job
    '''
    def __init__(self, svhn_loader_train, svhn_loader_test, opt):
        '''
        :param svhn_loader_train: loader of the SVHN train dataset
        :param svhn_loader_test: loader of the SVHN test dataset
        :param opt: options contain parameters for training
        '''
        self.opt = opt

        self.num_classes = 10
        self.nc = 3
        self.use_gpu = True if torch.cuda.is_available() else False
        self.best_netG_filename = '{}/netG_best.pth'.format(self.opt.out_dir)
        self.best_netD_filename = '{}/netD_best.pth'.format(self.opt.out_dir)

        self.svhn_loader_train = svhn_loader_train
        self.svhn_loader_test = svhn_loader_test

        self.netG, self.netD = self._build_model()
        self.g_optimizer, self.d_optimizer = self._create_optimizers()
        # , self.g_lr_scheduler, self.d_lr_scheduler

    def _build_model(self):
        '''
        Builds generator and discriminator
        :return: tuple of (generator, discriminator)
        '''
        netG = _netG(
            self.opt.nz, self.opt.ngf, self.opt.alpha,
            self.nc, self.use_gpu)
        netG.apply(self._weights_init)
        print(netG)

        netD = _netD(
            self.opt.ndf, self.opt.alpha, self.nc,
            self.opt.drop_rate, self.num_classes, self.use_gpu)
        netD.apply(self._weights_init)
        print(netD)

        if self.use_gpu:
            netG = netG.cuda()
            netD = netD.cuda()

        return netG, netD

    def _weights_init(self, module):
        '''
        Initializes weights for generator and discriminator
        :param module: generator or discriminator network expected
        '''
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _create_optimizers(self):
        '''
        Creates optimizers
        :return: tuple of (generator optimizer, discriminator optimizer)
        '''
        g_params = list(self.netG.parameters())
        d_params = list(self.netD.parameters())

        g_optimizer = optim.Adam(g_params, self.opt.learning_rate,
                                 betas=(self.opt.beta1, 0.999))
        d_optimizer = optim.Adam(d_params, self.opt.learning_rate,
                                 betas=(self.opt.beta1, 0.999))
        # g_lr_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=0.9)
        # d_lr_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=0.9)

        return g_optimizer, d_optimizer #, g_lr_scheduler, d_lr_scheduler

    def _to_var(self, x):
        '''
        Creates a variable for a tensor
        :param x: PyTorch Tensor
        :return: Variable that wraps the tensor
        '''
        if self.use_gpu:
            x = x.cuda()
        return Variable(x)

    def _one_hot(self, x):
        '''
        One-hot encoding of the vector of classes. It uses number of classes + 1 to
        encode fake images
        :param x: vector of output classes to one-hot encode
        :return: one-hot encoded version of the input vector
        '''
        label_numpy = x.data.cpu().numpy()
        label_onehot = np.zeros((label_numpy.shape[0], self.num_classes + 1))
        label_onehot[np.arange(label_numpy.shape[0]), label_numpy] = 1
        label_onehot = self._to_var(torch.FloatTensor(label_onehot))
        return label_onehot

    def _reset_grad(self):
        '''
        Reset gradients for generator and discriminator optimizers
        '''
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def _create_out_dir(self):
        '''
        Creates output directory
        '''
        if not os.path.exists(self.opt.out_dir):
            os.makedirs(self.opt.out_dir)

    def load_model(self, netG_filename, netD_filename):
        '''
        Loads generator and discriminator models from the disk
        :param netG_filename: file that stores generator weights
        :param netD_filename: file that stores discriminator weights
        '''
        self.netG.load_state_dict(torch.load(netG_filename))
        self.netD.load_state_dict(torch.load(netD_filename))

    def test(self, epoch_num, epochs):
        '''
        Runs predictions in the discriminator network on test images
        :param epoch_num: epoch number. Used only for printing
        :param epochs: total number of epochs. Used only for printing
        '''
        correct = 0
        num_samples = 0
        self.netD.eval()

        for i, data in enumerate(self.svhn_loader_test):
            # load the test data
            svhn_data, svhn_labels = data
            svhn_data = self._to_var(svhn_data)
            svhn_labels = self._to_var(svhn_labels).long().squeeze()

            # run the prediction
            d_out, _, _, _ = self.netD(svhn_data)

            # calculate number of correctly predicted numbers
            _, pred_idx = torch.max(d_out.data, 1)
            eq = torch.eq(svhn_labels.data, pred_idx)
            correct += torch.sum(eq.float())
            num_samples += len(svhn_labels)

            if i % 50 == 0:
                print('Test:\tepoch {}/{}\tsamples {}/{}'.format(
                    epoch_num, epochs, i + 1, len(self.svhn_loader_test)))

        accuracy = correct/max(1.0, 1.0 * num_samples)
        print('Test:\tepoch {}/{}\taccuracy {}'.format(epoch_num, epochs, accuracy))

    def train(self):
        self._create_out_dir()

        # initialize variables
        d_gan_criterion = nn.BCEWithLogitsLoss()
        noise = torch.FloatTensor(self.opt.batch_size, self.opt.nz, 1, 1)
        fixed_noise = torch.FloatTensor(self.opt.batch_size,
                                        self.opt.nz, 1, 1).normal_(0, 1)
        fixed_noise = self._to_var(fixed_noise)
        d_gan_labels_real = torch.LongTensor(self.opt.batch_size)
        d_gan_labels_fake = torch.LongTensor(self.opt.batch_size)
        best_accuracy = 0.0

        # iterate over all epochs
        for epoch in range(1, self.opt.epochs + 1):
            masked_correct = 0
            num_samples = 0

            # train on SVHN images
            self.netD.train()
            self.netG.train()
            for i, data in enumerate(self.svhn_loader_train):
                svhn_data, svhn_labels, label_mask = data
                svhn_data = self._to_var(svhn_data)
                svhn_labels = self._to_var(svhn_labels).long().squeeze()
                label_mask = self._to_var(label_mask).float().squeeze()

                # -------------- train netD --------------

                self._reset_grad()

                # train with real images
                # d_out == softmax(d_class_logits)
                d_out, d_class_logits_on_data, d_gan_logits_real, d_sample_features = self.netD(svhn_data)
                d_gan_labels_real.resize_as_(svhn_labels.data.cpu()).fill_(1)
                d_gan_labels_real_var = self._to_var(d_gan_labels_real).float()
                d_gan_loss_real = d_gan_criterion(
                    d_gan_logits_real,
                    d_gan_labels_real_var)

                # train with fake images
                noise.resize_(svhn_labels.data.shape[0], self.opt.nz, 1, 1).normal_(0, 1)
                noise_var = self._to_var(noise)
                fake = self.netG(noise_var)

                # call detach() to avoid backprop for netG here
                _, _, d_gan_logits_fake, _ = self.netD(fake.detach())
                d_gan_labels_fake.resize_(svhn_labels.data.shape[0]).fill_(0)
                d_gan_labels_fake_var = self._to_var(d_gan_labels_fake).float()
                d_gan_loss_fake = d_gan_criterion(
                    d_gan_logits_fake,
                    d_gan_labels_fake_var)

                # total gan loss
                d_gan_loss = d_gan_loss_real + d_gan_loss_fake

                # d_out == softmax(d_class_logits)
                # see https://stackoverflow.com/questions/34240703/whats-the-difference-between-softmax-and-softmax-cross-entropy-with-logits/39499486#39499486
                svhn_labels_one_hot = self._one_hot(svhn_labels)
                d_class_loss_entropy = -torch.sum(svhn_labels_one_hot * torch.log(d_out), dim=1)

                d_class_loss_entropy = d_class_loss_entropy.squeeze()
                delim = torch.max(torch.Tensor([1.0, torch.sum(label_mask.data)]))
                d_class_loss = torch.sum(label_mask * d_class_loss_entropy) / delim
                # numpy_labels = svhn_labels.data.cpu().numpy()
                # delim = torch.Tensor(numpy_labels.shape[0])
                # d_class_loss = torch.sum(d_class_loss_entropy) / delim # DO THIS FOR QUICK TEST ONLY!!!
                
                # total discriminator loss
                d_loss = d_gan_loss + d_class_loss

                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                # -------------- update netG --------------

                self._reset_grad()

                # call netD again to do backprop for netG here
                _, _, _, d_data_features = self.netD(fake)

                # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
                # This loss consists of minimizing the absolute difference between the expected features
                # on the data and the expected features on the generated samples.
                # This loss works better for semi-supervised learning than the tradition GAN losses.
                data_features_mean = torch.mean(d_data_features, dim=0).squeeze()
                sample_features_mean = torch.mean(d_sample_features, dim=0).squeeze()

                g_loss = torch.mean(torch.abs(data_features_mean - sample_features_mean))

                g_loss.backward()
                self.g_optimizer.step()

                _, pred_class = torch.max(d_class_logits_on_data, 1)
                eq = torch.eq(svhn_labels, pred_class)
                masked_correct += torch.sum(label_mask * eq.float())
                # correct = torch.sum(eq.float())
                # masked_correct += correct
                num_samples += torch.sum(label_mask)
                # num_samples += numpy_labels.shape[0]

                if i % 200 == 0:
                    print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tsamples {}/{}'.
                          format(epoch, self.opt.epochs,
                                 d_gan_loss.data[0], d_class_loss.data[0],
                                 g_loss.data[0], i + 1,
                                 len(self.svhn_loader_train)))
                    real_cpu, _, _ = data
                    vutils.save_image(real_cpu,
                                      '{}/real_samples.png'.format(self.opt.out_dir),
                                      normalize=True)
                    fake = self.netG(fixed_noise)
                    vutils.save_image(fake.data,
                                      '{}/fake_samples_epoch_{:03d}.png'.format(self.opt.out_dir, epoch),
                                      normalize=True)

            accuracy = masked_correct.data[0]/max(1.0, num_samples.data[0])
            print('Training:\tepoch {}/{}\taccuracy {}'.format(epoch, self.opt.epochs, accuracy))

            self.test(epoch, self.opt.epochs)

            # do checkpointing
            netG_filename = '{}/netG_epoch_{}.pth'.format(self.opt.out_dir, epoch)
            netD_filename = '{}/netD_epoch_{}.pth'.format(self.opt.out_dir, epoch)
            torch.save(self.netG.state_dict(), netG_filename)
            torch.save(self.netD.state_dict(), netD_filename)

            # decay learning rate
            # self.g_lr_scheduler.step()
            # self.d_lr_scheduler.step()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                shutil.copyfile(netG_filename, self.best_netG_filename)
                shutil.copyfile(netD_filename, self.best_netD_filename)
