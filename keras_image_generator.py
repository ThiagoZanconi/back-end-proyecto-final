import keras as keras
from keras import layers, models
from matplotlib import pyplot as plt
def build_autoencoder():
    # Codificador
    encoder = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2))
    ])

    # Decodificador
    decoder = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(16, 16, 64)),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')  # Output RGB
    ])

    # Autoencoder completo
    input_img = layers.Input(shape=(64, 64, 3))
    encoded = encoder(input_img)
    decoded = decoder(encoded)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


dataset = keras.preprocessing.image_dataset_from_directory(
    "resources/swords",
    labels=None,
    color_mode="rgb",
    image_size=(64, 64),
    batch_size=32,
    shuffle=True
)
dataset = dataset.map(lambda x: (x, x))
autoencoder = build_autoencoder()
autoencoder.fit(dataset, epochs=20)

for batch in dataset.take(1):
    originals = batch[0].numpy()  
    reconstructions = autoencoder.predict(originals)
    batch_size = originals.shape[0]  # cuántas imágenes hay en este batch
    for i in range(batch_size):
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(originals[i])
        plt.axis('off')
        plt.title("Original")

        plt.subplot(2, batch_size, i + 1 + batch_size)
        plt.imshow(reconstructions[i])
        plt.axis('off')
        plt.title("Reconstruida")

    plt.show()
