import tensorflow as tf
from tensorflow.keras import mixed_precision, Sequential
from tensorflow.keras.layers import (
    Lambda,
    Softmax,
    Conv2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.applications import ResNet50, DenseNet121
from repr_partitions_cluster.src.HDF5Generator import (
    HDF5Generator,
    HDF5GeneratorFilter,
    FilterMode,
)
import repr_partitions_cluster.src.vision_transformer as vision_transformer
from pathlib import Path
import wandb
from wandb.keras import WandbCallback
from repr_partitions_cluster.src.callbacks import ConfusionMatrix, Predictions

if __name__ == "__main__":
    root = Path(".") / "data"
    for model_name in ["resnet50"]:
            tf.keras.utils.set_random_seed(1)
            tf.random.set_seed(1)
            # tf.config.experimental.enable_op_determinism()
            config = {
                "batch_size": 32,  # [32,128,256]
                "num_epochs": 100,
                "lr": 1e-3,  # [1e-2,1e-3,1e-4,1e-5]
                "betas": (0.9, 0.99),
                "optimizer": "adam",
                "min_delta": 1e-3,
                "patience": 10,
                "network": model_name,  # ["resnet50","densenet121","dense"]
                "grid_size": 128,  # [64,128,256]
                "last_layers": "flatten_dense",  # ["flatten_dense","glob_avg"]
                "num_samples": 20000,
                "dropout": 0.2,  # [0.,0.1,0.2,0.5]
                "denses": [20, 10, 5, 2],  # [[2],[20,2],[20,10,2],[20,10,5,2]],
                "transfer": False,
                "num_pts": 1000,
                
                
                "patch_size": 8,
                "projection_dim": 64,
                "num_heads": 8,
                "transformer_units": [128, 64],
                "transformer_layers": 8,
                "mlp_head_units":[2048, 1024],
                "dataset": "diversified_examples"
            }
            wandb.init(
                config=config,
                project="Recherche Maitrise",  # Title of your project
                group="Top view image",  # In what group of runs do you want this run to be in?
                name=f"{config['network']} - head test {config['num_heads']}",
                tags=["top_view_image", "learning_rate"],
                save_code=True,
                entity="romo-1245",
            )
            path_caches = [
                root
                / "image_dataset"
                / f"dataset_ia_2_clusters_{config['num_pts']}pts_grid_{config['grid_size']}px_{dataset}.hdf5"
                for dataset in ["tr", "val"]
            ]
            ds = {
                dataset: tf.data.Dataset.from_generator(
                    HDF5Generator(path),
                    output_signature=(
                        tf.TensorSpec(shape=(config["grid_size"], config["grid_size"], 2), dtype=tf.float32),  # type: ignore
                        tf.TensorSpec(shape=(2,), dtype=tf.float32),  # type: ignore
                    ),
                )
                for dataset, path in zip(["tr", "val"], path_caches)
            }
            path_cache = (
                root
                / "image_dataset"
                / f"dataset_ia_2_clusters_{config['num_pts']}pts_grid_{config['grid_size']}px_val.hdf5"
            )
            path_metadata = root / f"dataset_{config['num_pts']}pts_40ksamples.hdf5"
            for filter_mode in FilterMode:
                ds["val_" + filter_mode.value] = tf.data.Dataset.from_generator(
                    HDF5GeneratorFilter(path_cache, path_metadata, filter_mode),
                    output_signature=(
                        tf.TensorSpec(shape=(config["grid_size"], config["grid_size"], 2), dtype=tf.float32),  # type: ignore
                        tf.TensorSpec(shape=(2,), dtype=tf.float32),  # type: ignore
                    ),
                )
            preprocessing = Sequential(
                [
                    Lambda(
                        lambda x: x / 5.0,
                        # input_shape=(config["grid_size"], config["grid_size"], 2),
                        name="normalize",
                    )
                ]
            )
            for k in ds.keys():
                ds[k] = (
                    ds[k]
                    .map(
                        lambda x, y: (preprocessing(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                    .shuffle(5456)
                    .batch(config["batch_size"])
                    .prefetch(tf.data.experimental.AUTOTUNE)
                )
            end_model_layers = None
            if config["network"] == "vision_transformer":
                end_model_layers = []
            elif config["last_layers"] == "flatten_dense" and config["network"] != "vision_transformer":
                end_model_layers = [Flatten(), Dense(2)]
            elif config["last_layers"] == "glob_avg" and config["network"] != "vision_transformer":
                end_model_layers = [
                    Conv2D(2, (1, 1), padding="same"),
                    GlobalAveragePooling2D(),
                ]
            elif config["last_layers"] == "flatten_dense_drop" and config["network"] != "vision_transformer":
                end_model_layers = [Flatten()]
                for i, filters in enumerate(config["denses"]):
                    end_model_layers.append(Dropout(config["dropout"]))
                    end_model_layers.append(
                        Dense(
                            filters,
                            activation="relu"
                            if i < len(config["denses"]) - 1
                            else "linear",
                        )
                    )
            else:
                raise Exception
            base_network = None
            if config["network"] == "resnet50":
                base_network = ResNet50(
                    include_top=False,
                    input_shape=(config["grid_size"], config["grid_size"], 3),
                    pooling=None,
                    weights="imagenet" if config["transfer"] else None,
                )
            elif config["network"] == "densenet121":
                base_network = DenseNet121(
                    include_top=False,
                    input_shape=(config["grid_size"], config["grid_size"], 3),
                    pooling=None,
                    weights="imagenet" if config["transfer"] else None,
                )
            elif config["network"] == "vision_transformer":
                base_network = vision_transformer.create_model(
                    input_shape=(config["grid_size"], config["grid_size"], 3),
                    patch_size=config["patch_size"],
                    num_patches=(config["grid_size"] // config["patch_size"]) ** 2,
                    projection_dim=config["projection_dim"],
                    transformer_layers=config["transformer_layers"],
                    num_heads=config["num_heads"],
                    transformer_units=config["transformer_units"],
                    mlp_head_units=config["mlp_head_units"],
                    num_classes=2,
                )
            else:
                raise Exception
            model = Sequential(
                [
                    Conv2D(
                        3,
                        (1, 1),
                        padding="same",
                        input_shape=(config["grid_size"], config["grid_size"], 2),
                    ),
                    base_network,
                    *end_model_layers,
                    Softmax(),
                ]
            )
            print(model.summary())

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=config["lr"],
                    beta_1=config["betas"][0],
                    beta_2=config["betas"][1],
                ),
                loss="categorical_crossentropy",
                metrics=["MAE", "accuracy"],
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
                WandbCallback(
                    validation_data=ds["val"], predictions=3, save_model=False
                ),
                ConfusionMatrix(
                    validation_data=ds["val"],
                    classes_names=("BI", "FI"),
                    name="full_validation",
                ),
                ConfusionMatrix(
                    validation_data=ds["val_random"],
                    classes_names=("BI", "FI"),
                    name="random_validation",
                ),
                ConfusionMatrix(
                    validation_data=ds["val_kmeans+"],
                    classes_names=("BI", "FI"),
                    name="kmeans_validation",
                ),
                Predictions(
                    validation_path=path_caches[1],
                    path_metadata=path_metadata,
                    classes_names=("BI", "FI"),
                    num_pred=5,
                ),
            ]
            print(tf.config.list_physical_devices("GPU"))
            model.fit(
                ds["tr"],
                epochs=config["num_epochs"],
                validation_data=ds["val"],
                callbacks=callbacks,
            )
            [loss, *metrics] = model.evaluate(ds["val"])
            wandb.log(
                {
                    "val_final_loss": loss,
                    "val_final_MAE": metrics[0],
                    "val_final_accuracy": metrics[1],
                }
            )
            [loss, *metrics] = model.evaluate(ds["val_random"])
            wandb.log(
                {
                    "val_random_final_loss": loss,
                    "val_random_final_MAE": metrics[0],
                    "val_random_final_accuracy": metrics[1],
                }
            )
            [loss, *metrics] = model.evaluate(ds["val_kmeans+"])
            wandb.log(
                {
                    "val_kmeans+_final_loss": loss,
                    "val_kmeans+_final_MAE": metrics[0],
                    "val_kmeans+_final_accuracy": metrics[1],
                }
            )
            wandb.finish()
