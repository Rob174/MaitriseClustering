import tensorflow as tf
from tensorflow.keras import mixed_precision, Sequential
from tensorflow.keras.layers import Lambda, Softmax, Conv2D, GlobalAveragePooling2D,Flatten,Dense
from tensorflow.keras.applications import ResNet50,DenseNet121
from repr_partitions_cluster.src.HDF5Generator import HDF5Generator
from pathlib import Path
import wandb
from wandb.keras import WandbCallback
from repr_partitions_cluster.src.callbacks import ConfusionMatrix

if __name__ == "__main__":
    root = Path(".")
    config = {
        "batch_size": 32,
        "num_epochs": 1,
        "lr": 1e-3,
        "betas": (0.9, 0.99),
        "optimizer": "adam",
        "min_delta": 1e-3,
        "patience": 10,
        "network": "densenet121",
        "grid_size": 128,
        "last_layers":"flatten_dense"
    }
    wandb.init(
        config=config,
        project="Recherche Maitrise",  # Title of your project
        group="Top view image",  # In what group of runs do you want this run to be in?
        name="DenseNet121",
        save_code=True,
        entity="romo-1245"
    )
    ds = {
        dataset: tf.data.Dataset.from_generator(
            HDF5Generator(
                    root
                    / "data"
                    / "image_dataset"
                    / f"dataset_ia_2_clusters_grid_{config['grid_size']}px_{dataset}.hdf5"
                ),
            output_signature=(
                tf.TensorSpec(shape=(config["grid_size"], config["grid_size"], 2), dtype=tf.float32),  # type: ignore
                tf.TensorSpec(shape=(2,), dtype=tf.float32),  # type: ignore
            ),
        )
        for dataset in ["tr", "val"]
    }
    preprocessing = Sequential(
        [
            Lambda(
                lambda x: x / 5.0,
                # input_shape=(config["grid_size"], config["grid_size"], 2),
            )
        ]
    )
    ds["tr"] = (
        ds["tr"]
        .map(lambda x, y: (preprocessing(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(5456)
        .batch(config["batch_size"])
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds["val"] = (
        ds["val"]
        .map(lambda x, y: (preprocessing(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1364)
        .batch(config["batch_size"])
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    end_model_layers = None
    if config["last_layers"] == "flatten_dense":
        end_model_layers = [Flatten(),Dense(2)]
    elif config["last_layers"] == "glob_avg":
        end_model_layers = [Conv2D(2, (1, 1), padding="same"),GlobalAveragePooling2D()]
    model = Sequential(
        [
            Conv2D(
                3,
                (1, 1),
                padding="same",
                input_shape=(config["grid_size"], config["grid_size"], 2),
            ),
            DenseNet121(
                input_shape=(config["grid_size"], config["grid_size"], 3),
                include_top=False,
                pooling=None,
            ),
            *end_model_layers,
            Softmax(),
        ]
    )

    # def accuracy_fn(y_true, y_pred):
    #     return 1 - tf.reduce_sum(
    #         tf.cast(tf.math.count_nonzero(y_true - y_pred), dtype=tf.float32)
    #     ) / tf.cast(tf.size(y_pred), dtype=tf.float32)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["lr"],
            beta_1=config["betas"][0],
            beta_2=config["betas"][1],
        ),
        loss="categorical_crossentropy",
        metrics=["MAE","accuracy"],
        run_eagerly=True,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=config["min_delta"],
            patience=config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        WandbCallback(validation_data=ds["val"],predictions=3,save_model=False),
        ConfusionMatrix(validation_data=ds["val"],classes_names=("BI","FI")),
    ]
    print(tf.config.list_physical_devices("GPU"))
    model.fit(
        ds["tr"],
        epochs=config["num_epochs"],
        validation_data=ds["val"],
        callbacks=callbacks,
    )
    wandb.finish()
