from keras.layers import Input
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

class GAN():
    def __init__(self, discriminator, generator):

        self.discriminator = discriminator
        self.generator = generator
        
    def build_networks(self, optimizer, loss, metrics, gen_input = Input(shape=(100,))):
        self.discriminator.compile(loss=loss, 
            optimizer=optimizer,
            metrics=metrics)
        self.generator.compile(loss=loss, optimizer=optimizer)
        
        # The generator takes noise as input and generated imgs
        z = gen_input
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss=loss, optimizer=optimizer)

    def train(self, X_train, epochs, batch_size=128, save_interval=0):

      
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if save_interval != 0:
                if epoch % save_interval == 0:
                    self.save_imgs(epoch)
                    
    def create_fake(self, number_of_fake=10):
        noise = np.random.normal(0, 1, (number_of_fake, 100))
        return self.generator.predict(noise)

    def save_imgs(self, epoch, row=5, col=5):
        r, c = row, col
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan_%d.png" % epoch)
        plt.close()