import torch
import torch.nn as nn
from model import _netG, _netD
from torch import optim
from torch.autograd import Variable
import numpy as np

class Solver:
    def __init__(self, svhn_loader_train, svhn_loader_test, batch_size):
        self.nz = 100
        self.real_image_size = (3, 32, 32)
        self.lrelu_alpha = 1e-2
        self.drop_rate = .5
        self.g_size_mult = 32
        self.d_size_mult = 64
        self.num_classes = 10
        self.use_gpu = True if torch.cuda.is_available() else False
        self.learning_rate = 3e-3
        self.beta1 = .5
        self.svhn_loader_train = svhn_loader_train
        self.svhn_loader_test = svhn_loader_test
        self.epochs = 25
        self.batch_size = batch_size

        self.generator, self.discriminator = self._build_model()
        self.g_optimizer, self.d_optimizer = self._create_optimizers()

    def _build_model(self):
        generator = _netG(
            self.nz, self.g_size_mult, self.lrelu_alpha,
            self.real_image_size[0])
        generator.apply(self._weights_init)
        # TODO: load weights from file if it exists

        discriminator = _netD(
            self.d_size_mult, self.lrelu_alpha, self.real_image_size[0],
            self.drop_rate, self.num_classes)
        discriminator.apply(self._weights_init)
        # TODO: load weights from file if it exists

        if self.use_gpu:
            generator = generator.cuda()
            discriminator = discriminator.cuda()

        return generator, discriminator

    def _weights_init(self, module):
        '''
        Custom weights initialization called on generator and discriminator
        '''
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _create_optimizers(self):
        g_params = list(self.generator.parameters())
        d_params = list(self.discriminator.parameters())

        g_optimizer = optim.Adam(g_params, self.learning_rate)
        d_optimizer = optim.Adam(d_params, self.learning_rate)

        return g_optimizer, d_optimizer

    def _to_var(self, x):
        if self.use_gpu:
            x = x.cuda()
        return Variable(x)

    def _one_hot(self, x):
        ones = torch.sparse.torch.eye(self.num_classes)
        one_hot = ones.index_select(0, x.data)
        return Variable(one_hot)

#        indices = x.long().view(-1, 1)
#        one_hot = torch.zeros_like(x)
#        return one_hot.scatter(2, indices, 1)

    def train(self):
        svhn_iter = iter(self.svhn_loader_train)
        iter_per_epoch = len(svhn_iter)
        print(iter_per_epoch)

        d_gan_criterion = nn.BCEWithLogitsLoss()
        
        noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1)

        for epoch in range(1, self.epochs + 1):
            for _, data in enumerate(self.svhn_loader_train):
                # load svhn dataset
                svhn_data, svhn_labels, label_mask = data
                svhn_data = self._to_var(svhn_data)
                svhn_labels = self._to_var(svhn_labels).long().squeeze()
                label_mask = self._to_var(label_mask).float().squeeze()

                # -------------- train discriminator --------------

                # train with real images
                self.d_optimizer.zero_grad()

                # d_out == softmax(d_class_logits)
                d_out, d_class_logits_on_data, d_gan_logits_real, d_sample_features = self.discriminator(svhn_data)
                d_gan_labels_real = self._to_var(torch.ones_like(d_gan_logits_real.data))
                d_gan_loss_real = d_gan_criterion(
                    d_gan_logits_real,
                    d_gan_labels_real)

                # train with fake images
                noise.resize_(self.batch_size, self.nz, 1, 1).normal_(0, 1)
                noise_var = self._to_var(noise)
                fake = self.generator(noise_var)

                # call detach() to avoid backprop for generator here
                _, _, d_gan_logits_fake, _ = self.discriminator(fake.detach())

                d_gan_labels_fake = self._to_var(torch.zeros_like(d_gan_logits_fake.data))
                d_gan_loss_fake = d_gan_criterion(
                    d_gan_logits_fake,
                    d_gan_labels_fake)

                d_gan_loss = d_gan_loss_real + d_gan_loss_fake

                # d_out == softmax(d_class_logits)
                # see https://stackoverflow.com/questions/34240703/whats-the-difference-between-softmax-and-softmax-cross-entropy-with-logits/39499486#39499486
                svhn_labels_one_hot = self._one_hot(svhn_labels)
                d_class_loss_entropy = -torch.sum(svhn_labels_one_hot * torch.log(d_out), dim=1)
                
                # d_class_loss_entropy = d_class_criterion(
                #     d_class_logits_on_data,
                #     self._one_hot(svhn_labels)
                # )

                d_class_loss_entropy = d_class_loss_entropy.squeeze()
                delim = torch.max(torch.Tensor([1.0, torch.sum(label_mask.data)]))
                d_class_loss = torch.sum(label_mask * d_class_loss_entropy) / delim
                
                d_loss = d_gan_loss + d_class_loss
                d_loss.backward()
                self.d_optimizer.step()

                # -------------- update generator --------------
                
                self.g_optimizer.zero_grad()

                # call discriminator again to do backprop for generator here
                _, _, _, d_data_features = self.discriminator(fake)
                
                # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
                # This loss consists of minimizing the absolute difference between the expected features
                # on the data and the expected features on the generated samples.
                # This loss works better for semi-supervised learning than the tradition GAN losses.
                data_features_mean = torch.mean(d_data_features, dim=0)
                sample_features_mean = torch.mean(d_sample_features.detach(), dim=0)
                
                g_loss = torch.mean(torch.abs(data_features_mean - sample_features_mean))

                _, pred_class = torch.max(d_class_logits_on_data, 1)
                eq = torch.eq(svhn_labels, pred_class)
                correct = torch.sum(eq.float())
                masked_correct = torch.sum(label_mask * eq.float())

                g_loss.backward()
                self.g_optimizer.step()

                print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tmasked correct {}'.
                        format(epoch, self.epochs, d_gan_loss.data[0], d_class_loss.data[0], g_loss.data[0], masked_correct.data[0]))

            for _, data in enumerate(self.svhn_loader_test):
                # load svhn dataset
                svhn_data, svhn_labels = data
                svhn_data = self._to_var(svhn_data)
                svhn_labels = self._to_var(svhn_labels).long().squeeze()

                # -------------- train discriminator --------------

                # train with real images
                d_out, _, _, _ = self.discriminator(svhn_data)
                _, pred_idx = torch.max(d_out.data, 1)
                eq = torch.eq(svhn_labels, pred_idx)
                correct = torch.sum(eq.float())
                
                print('Test:\tepoch {}/{}\tcorrect {}'.format(epoch, self.epochs, correct))

            # TODO: save checkpoints and the best model weights
