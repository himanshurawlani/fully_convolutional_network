import tensorflow as tf

def export(input_h5_file, export_path):
    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = tf.keras.models.load_model(input_h5_file)
    model.save(export_path, save_format='tf')
    print(f"SavedModel created at {export_path}")

if __name__ == "__main__":

    input_h5_file = './snapshots/dense_model_epoch_48_loss_0.74_acc_0.71_val_loss_0.77_val_acc_0.70.h5'
    export_path = './flower_classifier/1'
    export(input_h5_file, export_path)
