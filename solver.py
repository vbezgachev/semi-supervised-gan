import torch
import torch.nn as nn
from model import _netG, _netD
from torch import optim
from torch.autograd import Variable
import numpy as np

class Solver:
    def __init__(self, svhn_loader, batch_size):
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
        self.svhn_loader = svhn_loader
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

        g_optimizer = optim.Adam(g_params, self.learning_rate, self.beta1)
        d_optimizer = optim.Adam(d_params, self.learning_rate, self.beta1)

        return g_optimizer, d_optimizer

    # def _define_criterion(self):
    #     d_gan_criterion = nn.BCEWithLogitsLoss()
    #     d_class_criterion = nn.MultiLabelSoftMarginLoss()
        # g_criterion: calculate using features, torch.mean() directly

    def _to_var(self, x):
        if self.use_gpu:
            x = x.cuda()
        return Variable(x)

    def train(self):
        svhn_iter = iter(self.svhn_loader)
        iter_per_epoch = len(svhn_iter)
        print(iter_per_epoch)

        d_gan_criterion = nn.BCEWithLogitsLoss()
        d_class_criterion = nn.MultiLabelSoftMarginLoss()

        label_mask = torch.zeros_like(iter_per_epoch)
        label_mask[0:1000] = 1
        label_mask = self._to_var(label_mask).long().squeeze()

        noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1)

        for step in range(1, self.epochs + 1):
            # reset data iterator for each epoch
            if step % iter_per_epoch == 0:
                svhn_iter = iter(self.svhn_loader)
                np.random.shuffle(label_mask)

            # load svhn dataset
            svhn_data, svhn_labels = svhn_iter.next()
            svhn_data = self._to_var(svhn_data)
            svhn_labels = self._to_var(svhn_labels).long().squeeze()

            # -------------- train discriminator --------------

            # train with real images
            self.d_optimizer.zero_grad()
            _, d_class_logits_on_data, d_gan_logits_real, d_sample_features = self.discriminator(svhn_data)
            d_gan_labels_real = self._to_var(torch.ones_like(d_gan_logits_real.data))
            d_gan_loss_real = torch.mean(
                d_gan_criterion(
                    d_gan_logits_real,
                    d_gan_labels_real
                ))

            # train with fake images
            noise.resize_(self.batch_size, self.nz, 1, 1).normal_(0, 1)
            noise_var = self._to_var(noise)
            fake = self.generator(noise_var)

            # call detach() to avoid backprop for generator here
            _, _, d_gan_logits_fake, _ = self.discriminator(fake.detach())

            d_gan_labels_fake = self._to_var(torch.zeros_like(d_gan_logits_fake.data))
            d_gan_loss_fake = torch.mean(
                d_gan_criterion(
                    d_gan_logits_fake,
                    d_gan_labels_fake
                ))

            d_gan_loss = d_gan_loss_real + d_gan_loss_fake

            d_class_loss_entropy = d_class_criterion(
                d_class_logits_on_data,
                svhn_labels
            )
            d_class_loss = torch.sum(label_mask * d_class_loss_entropy) / torch.max(1, torch.sum(label_mask))
            
            d_loss = d_gan_loss + d_class_loss
            d_loss.backward()
            self.d_optimizer.step()

            # -------------- update generator --------------
            
            self.g_optimizer.zero_grad()

            # call discriminator again to do backprop for generator here
            _, _, _, d_features_on_data = self.discriminator(fake)
            
            # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
            # This loss consists of minimizing the absolute difference between the expected features
            # on the data and the expected features on the generated samples.
            # This loss works better for semi-supervised learning than the tradition GAN losses.
            data_features_mean = torch.mean(d_features_on_data, dim=0)
            sample_features_mean = torch.mean(d_sample_features, dim=0)
            
            g_loss = torch.mean(torch.abs(data_features_mean - sample_features_mean))

            pred_class = torch.max(d_class_logits_on_data, 1).long()
            eq = torch.eq(svhn_labels, pred_class)
            correct = torch.sum(eq.float())
            masked_correct = torch.sum(label_mask * correct)

            print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tmasked correct {}'.
                      format(step, self.epochs, d_gan_loss, d_class_loss, g_loss, masked_correct))
