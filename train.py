import tensorflow as tf
from utils import split_dataset, get_dataset_stats
from model import FCN_model
from generator import Generator

def train(model, train_generator, val_generator):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    epochs = 50

    checkpoint_path = './snapshots'
    history = model.fit_generator(generator=train_generator,
                                    epochs=epochs,
                                    callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1, save_format='tf')],
                                    validation_data=val_generator)

    return history

if __name__ == "__main__":
    # split_dataset()
    # get_dataset_stats()
    model = FCN_model()
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    train_generator = Generator(train_dir)
    val_generator = Generator(val_dir)
    history = train(model, train_generator, val_generator)