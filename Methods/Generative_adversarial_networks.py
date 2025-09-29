import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Reload the preprocessed dataset
df = pd.read_excel('C:/Users/yousf/OneDrive/Desktop/Preprocessed_Perovskite_Dataset.xlsx')

# Convert the DataFrame to a NumPy array for GAN processing
data = df.to_numpy()

# Data dimensions
data_dim = data.shape[1]

# GAN parameters
latent_dim = 100  # Size of the latent space

# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_dim=latent_dim),
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(data_dim, activation="linear")
    ])
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu", input_dim=data_dim),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Compile the GAN
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Combine generator and discriminator to create the GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002),
            loss='binary_crossentropy')

# Training GAN
def train_gan(data, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {epochs}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

# Train GAN on the dataset
train_gan(data, epochs=1000, batch_size=32)

# Generate new synthetic data
num_samples = 100
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data = generator.predict(noise)

# Save synthetic data to Excel
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
output_synthetic_path = 'C:/Users/yousf/Macquarie University/Jincheol Kim - 2024_PACE/00_Project without image data/Data/P.xlsx'
synthetic_df.to_excel(output_synthetic_path, index=False)

output_synthetic_path
