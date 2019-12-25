import tensorflow as tf

def export(input_h5_file, export_path):
    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = tf.keras.models.load_model(input_h5_file)
    model.save(export_path, save_format='tf')
    print(f"SavedModel created at {export_path}")

if __name__ == "__main__":

    input_h5_file = './snapshots/model_epoch_28_loss_0.99_acc_0.61_val_loss_0.98_val_acc_0.62.h5'
    export_path = './flower_classifier/1'
    export(input_h5_file, export_path)
