from pickletools import optimize
from xml.etree.ElementInclude import include
import tensorflow as tf
from tensorflow.keras import mixed_precision, Sequential
from tensorflow.keras.layers import Lambda, Softmax,Conv2D
from tensorflow.keras.applications import ResNet50
from repr_partitions_cluster.src.HDF5Generator import HDF5Generator
from pathlib import Path
import wandb
from wandb.keras import WandbCallback

if __name__ == "__main__":
    root = Path(".")
    config = {
        "batch_size": 32,
        "num_epochs": 100,
        "lr": 1e-3,
        "betas": (0.9, 0.99),
        "optimizer": "adam",
        "min_delta": 1e-3,
        "patience": 10,
        "network": "resnet50",
        "grid_size": 128,
    }
    wandb.init(
        config=config,
        project="Recherche Maitrise",  # Title of your project
        group="Top view image",  # In what group of runs do you want this run to be in?
        name="Base Model - Transfer learning",
        save_code=True,
    )
    ds = {
        dataset: tf.data.Dataset.from_tensor_slices(
            [
                str(
                    root
                    / "data"
                    / f"dataset_ia_2_clusters_grid_{config['grid_size']}px_{dataset}.hdf5"
                )
            ]
        ).interleave(
            lambda filename: tf.data.Dataset.from_generator(
                HDF5Generator(),
                output_signature=(
                    tf.TensorSpec(shape=(config["grid_size"], config["grid_size"], 2), dtype=tf.float32),  # type: ignore
                    tf.TensorSpec(shape=(2,), dtype=tf.float32),  # type: ignore
                ),
                args=(filename,),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
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
    model = Sequential(
        [
            Conv2D(3, (1,1), padding="same", input_shape=(config["grid_size"], config["grid_size"], 2)),
            ResNet50(
                input_shape=(config["grid_size"], config["grid_size"], 2),
                include_top=False,
                classes=2,
            ),
            Softmax(),
        ]
    )

    def accuracy_fn(y_true, y_pred):
        return 1 - tf.reduce_sum(
            tf.cast(tf.math.count_nonzero(y_true - y_pred), dtype=tf.float32)
        ) / tf.cast(tf.size(y_pred), dtype=tf.float32)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=config["lr"], beta_1=config["betas"][0], beta_2=config["betas"][1]
        ),
        loss="categorical_crossentropy",
        metrics=[accuracy_fn],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy_fn",
            min_delta=config["min_delta"][1],
            patience=config["patience"][1],
            restore_best_weights=True,
            verbose=1,
        ),
        WandbCallback(),
    ]
    model.fit(
        ds["tr"],
        epochs=config["num_epochs"],
        validation_data=ds["val"],
        callbacks=callbacks,
    )
