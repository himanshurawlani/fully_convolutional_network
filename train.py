import tensorflow as tf
from utils import split_dataset, get_dataset_stats
from model import FCN_model
from generator import Generator
import os

def train(model, train_generator, val_generator):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    epochs = 50

    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path, 'model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{acc:.2f}_val_loss_{val_loss:.2f}_val_acc_{val_acc:.2f}.h5')
    
    history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=len(train_generator),
                                    epochs=epochs,
                                    callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)],
                                    validation_data=val_generator,
                                    validation_steps=len(val_generator))

    return history

if __name__ == "__main__":
    
    model = FCN_model()
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    BATCH_SIZE=8
    train_generator = Generator(train_dir, BATCH_SIZE)
    val_generator = Generator(val_dir, BATCH_SIZE)
    history = train(model, train_generator, val_generator)
