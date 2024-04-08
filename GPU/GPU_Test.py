from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import keras_tuner as kt
import os


# KerasTuner model-building function
def build_model(hp):
    units = hp.Int(name="units", min_value=16, max_value=64, step=16)           # Sample hyperparameter values from the hp object. After sampling, these values
    model = keras.Sequential([                                                  # (such as the "units" variable here) are just regular Python constants.
        layers.Dense(units, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])         # Different kinds of hyperparameters are available: Int, Float, Boolean, Choice.
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model                                                                # The function returns a compiled model.


# KerasTuner HyperModel
class SimpleMLP(kt.HyperModel):
    def __init__(self, num_classes):                                            # Thanks to the object-oriented approach, we can configure model constants as
        self.num_classes = num_classes                                          # constructor arguments (instead of hardcoding them in the model-building function).

    def build(self, hp):                                                        # The build() method is identical to our prior  build_model() standalone function.
        units = hp.Int(name="units", min_value=16, max_value=64, step=16)
        model = keras.Sequential([
            layers.Dense(units, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        return model


def get_best_epoch(hp):
    model = build_model(hp)
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10)                    # Note the high patience value
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks)
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    return best_epoch


def get_best_trained_model(hp):
    best_epoch = get_best_epoch(hp)
    model = build_model(hp)
    model.fit(
        x_train_full, y_train_full,
        batch_size=128, epochs=int(best_epoch * 1.2))
    return model


if __name__ == "__main__":
    hypermodel = SimpleMLP(num_classes=10)

    # Available GPUs
    tf.config.list_physical_devices('GPU')

    tuner = kt.BayesianOptimization(
        build_model,                 # Specify the model-building function (or hyper-model instance).
        objective="val_accuracy",    # Specify the metric that the tuner will seek to optimize. Always specify validation metrics,
                                     # since the goal of the search process is to find models that generalize!
        max_trials=50,              # Maximum number of different model configurations (“trials”) to try before ending the search.
        executions_per_trial=2,      # To reduce metrics variance, you can train the same model multiple times and average the results.
                                     # executions_per_trial is how many training rounds (executions) to run for each model configuration (trial).
        directory="mnist_kt_test",   # Where to store the search logs.
        overwrite=True,              # Whether to overwrite data in directory to start a new search. Set this to True if you’ve modified
                                     # the model-building function, or to False to resume a previously started search with the same
                                     # model-building function.
    )

    if os.environ.get('TF_ENABLE_ONEDNN_OPTS') == '0':
        print("TF_ENABLE_ONEDNN_OPTS is set to 0. oneDNN custom operations are disabled.")
    else:
        print("TF_ENABLE_ONEDNN_OPTS is not set to 0. oneDNN custom operations may be enabled.")

    # Display the search space summary
    tuner.search_space_summary()

    # Run all of the simulations
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255
    x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255
    x_train_full = x_train[:]  # Reserve for later
    y_train_full = y_train[:]  # Reserve for later
    num_val_samples = 10000
    x_train, x_val = x_train[:-num_val_samples], x_train[-num_val_samples:]  # Set aside as a validation set
    y_train, y_val = y_train[:-num_val_samples], y_train[-num_val_samples:]  # Set aside as a validation set
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5), # Use an EarlyStopping callback to stop training when you start overfitting
    ]
    tuner.search(  # This takes the same arguments as fit() (it simply passes them down to fit() for each new model).
        x_train, y_train,
        batch_size=32,
        epochs=50,  # Use a large number of epochs (you don’t know in advance how many epochs each model will need)
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    top_n = 4
    # Return a list of HyperParameter objects, which you can pass to the model-building function
    best_hps = tuner.get_best_hyperparameters(top_n)

    best_models = []
    for hp in best_hps:
        model = get_best_trained_model(hp)
        model.evaluate(x_test, y_test)
        best_models.append(model)

    best_models = tuner.get_best_models(top_n)

    # Create a checkpoint with write()
    ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
    path = ckpt.write('/tmp/my_checkpoint')

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(path).expect_partial()

    for model in best_models:
        print(model.evaluate(x_test, y_test))

