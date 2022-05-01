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
    Normalization,BatchNormalization
)
from tensorflow.keras.applications import ResNet50, DenseNet121, VGG16
from repr_partitions_cluster.src.HDF5Generator import (
    HDF5Generator,
    HDF5GeneratorFilter,
    FilterMode,
)
import repr_partitions_cluster.src.vision_transformer as vision_transformer
from pathlib import Path
import wandb
from wandb.keras import WandbCallback
from repr_partitions_cluster.src.callbacks import (
    ConfusionMatrix,
    Predictions,
    LabelMaker,
    OneHotLabelMaker,
    EvaluateIntermediate,
    Scheduler,
)
from classification_models.keras import Classifiers
if __name__ == "__main__":
    root = Path(".") / "data"
    seed = 1
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
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
        "last_layers": "flatten_dense",  
        "num_samples": 20000,
        "dropout": 0.2,  
        "denses": [20, 10, 5, 2],  
        "transfer": False,
        "num_pts": 1000,
        "network_mode": "classification",
        "l2_reg": 0.0,
        "l2_weight_only": False,
        "patch_size": 8,
        "projection_dim": 64,
        "num_heads": 4,
        "transformer_units": [128, 64],
        "transformer_layers": 4,
        "mlp_head_units": [2048, 1024],
        "dataset": "",
        "seed": seed,
        "activation": "relu",
        "training_dataset":"full"
    }
    wandb.init(
        config=config,
        project="Recherche Maitrise",  # Title of your project
        group="Top view image",  # In what group of runs do you want this run to be in?
        name=f"{config['network']} - training dataset {config['training_dataset']} only lr {config['lr']}",  # Name of this run
        tags=["top_view_image"],
        save_code=True,
        entity="romo-1245",
    )
    path_prefix = ""
    num_classes = 2
    loss = "categorical_crossentropy"
    if config["network_mode"] == "regression":
        path_prefix = "_continuous"
        num_classes = 1
        loss = "MSE"
        
    # Create datasets
    path_caches = [
        root
        / "image_dataset"
        / f"dataset_ia_2_clusters_{config['num_pts']}pts{config['dataset']}{path_prefix}_grid_{config['grid_size']}px_{dataset}.hdf5"
        for dataset in ["tr", "val"]
    ]
    ds = {
        dataset: tf.data.Dataset.from_generator(
            HDF5Generator(path),
            output_signature=(
                tf.TensorSpec(shape=(config["grid_size"], config["grid_size"], 2), dtype=tf.float32),  # type: ignore
                tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),  # type: ignore
            ),
        )
        for dataset, path in zip(["tr", "val"], path_caches)
    }
    path_metadata = (
        root
        / f"dataset_{config['num_pts']}pts_40ksamples{config['dataset']}{config['dataset']}.hdf5"
    )
    for i,dataset in enumerate(["tr", "val"]):
        for filter_mode in FilterMode:
            ds[f"{dataset}_{filter_mode.value}"] = tf.data.Dataset.from_generator(
                HDF5GeneratorFilter(path_caches[i], path_metadata, filter_mode),
                output_signature=(
                    tf.TensorSpec(shape=(config["grid_size"], config["grid_size"], 2), dtype=tf.float32),  # type: ignore
                    tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),  # type: ignore
                ),
            )
    # Preprocess datasets, shuffle...
    preprocessing = Sequential(
        [
            # normalization
            Lambda(
                lambda x: x / 5.0,
                name="normalize",
            )
        ]
    )
    len_datasets = {}
    for k in ds.keys():
        ds[k] = (
            ds[k]
            .map(
                lambda x, y: (preprocessing(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .cache(f"cache_{k}")
            .shuffle(5456)
            .batch(config["batch_size"])
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        len_datasets[k] = 0
        for _ in ds[k]:
            len_datasets[k] += 1
    # Select dataset used for training depending of the initialization
    tr_dataset = None
    if config["training_dataset"] ==  "full":
        tr_dataset = ds["tr"]
    elif config["training_dataset"] == "random":
        tr_dataset = ds["tr_random"]
    elif config["training_dataset"] == "kmeans+":
        tr_dataset = ds["tr_kmeans+"]
    else:
        raise Exception

    # Create model
    ## Choose last layer
    end_model_layers = None
    if config["network"] == "vision_transformer":
        end_model_layers = [] # Vision transformer has already a flatten dense end
    elif (
        config["last_layers"] == "flatten_dense"
        and config["network"] != "vision_transformer"
    ):
        end_model_layers = [Flatten(), Dense(num_classes)]
    elif (
        config["last_layers"] == "glob_avg"
        and config["network"] != "vision_transformer"
    ):
        end_model_layers = [
            Conv2D(num_classes, (1, 1), padding="same"),
            GlobalAveragePooling2D(),
        ]
    elif (
        config["last_layers"] == "flatten_dense_drop"
        and config["network"] != "vision_transformer"
    ):
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
    ## Last layer softmax if classification task else linear if regression task 
    ## (we cannot put renormalize by the maximum difference as 
    ## we do not know the maximum difference possible)
    if config["network_mode"] != "regression":
        end_model_layers.append(Softmax())
    ## Create model backbone
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
            num_classes=num_classes,
        )
    elif config["network"] == "vgg16":
        base_network = VGG16(
            include_top=False,
            input_shape=(config["grid_size"], config["grid_size"], 3),
            pooling=None,
            weights="imagenet" if config["transfer"] else None,
        )
    elif config["network"] == "resnet18":
        ResNet18,_ = Classifiers.get('resnet18')
        base_network = ResNet18((config["grid_size"], config["grid_size"], 3), weights=None,include_top=False)
            
    else:
        raise Exception
    ## Regularization
    alpha = config["l2_reg"]
    if alpha > 0:
        for layer in base_network.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(
                layer, tf.keras.layers.Dense
            ):
                layer.kernel_regularizer = tf.keras.regularizers.l2(alpha)
                if (
                    len(layer.trainable_variables) > 1
                    and config["l2_weight_only"] is False
                ):
                    layer.bias_regularizer = tf.keras.regularizers.l2(alpha)
    ## leaky relu activation replacement
    if config['activation'] == 'lrelu':
        for l in base_network.layers:
            if isinstance(l, tf.keras.layers.Conv2D):
                l.activation = tf.keras.layers.LeakyReLU()
            if isinstance(l, tf.keras.layers.Activation):
                l.activation = tf.keras.layers.LeakyReLU()
    model = Sequential(
        [
            Conv2D( # To map from 2 filters (2 clusters) to 3 filters (for conventionnal image)
                3,
                (1, 1),
                padding="same",
                input_shape=(config["grid_size"], config["grid_size"], 2),
            ),
            base_network,
            *end_model_layers,
        ]
    )
    print(model.summary())
    # Metrics, callbacks ...
    metrics = ["MAE"]
    if config["network_mode"] == "regression":
        label_maker = OneHotLabelMaker() # Can transform continuous prediction into label best algorithm
        monitor = "val_accuracy"

        def accuracy(y_true, y_pred): # Build similar classification accuracy for comparison purposes
            true = tf.math.sign(y_true)
            pred = tf.math.sign(y_pred)
            return tf.reduce_mean(tf.cast(tf.equal(true, pred), tf.float32))

        metrics.append(accuracy)
    elif config["network_mode"] == "classification":
        label_maker = LabelMaker()
        metrics.append("accuracy")
        monitor = "val_accuracy"
    else:
        raise Exception
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["lr"],
            beta_1=config["betas"][0],
            beta_2=config["betas"][1],
        ),
        loss=loss,
        metrics=metrics,
        run_eagerly=True,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=config["min_delta"],
            patience=config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        WandbCallback(validation_data=ds["val"], predictions=3, save_model=False),
        ConfusionMatrix(
            data=ds["val"],
            classes_names=("BI", "FI"),
            name="full_validation",
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
        ConfusionMatrix(
            data=ds["val_random"],
            classes_names=("BI", "FI"),
            name="random_validation",
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
        ConfusionMatrix(
            data=ds["val_kmeans+"],
            classes_names=("BI", "FI"),
            name="kmeans_validation",
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
        ConfusionMatrix(
            data=ds["tr"],
            classes_names=("BI", "FI"),
            name="full_train",
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
        ConfusionMatrix(
            data=ds["tr_random"],
            classes_names=("BI", "FI"),
            name="random_train",
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
        ConfusionMatrix(
            data=ds["tr_kmeans+"],
            classes_names=("BI", "FI"),
            name="kmeans_train",
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
        Predictions(
            name="validation",
            validation_path=path_caches[1],
            path_metadata=path_metadata,
            classes_names=("BI", "FI"),
            num_pred=5,
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
            preprocessing=preprocessing,
        ),
        EvaluateIntermediate(
            data=ds[ # To evaluate the correct validation dataset if the model is not trained on the full dataset
                "val" if config["training_dataset"] == "full" 
                    else "val_"+config["training_dataset"]
                    ],
            name="validation" if config["training_dataset"] == "full" else "validation_"+config["training_dataset"],
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=1, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
        Predictions(
            name="train",
            validation_path=path_caches[0],
            path_metadata=path_metadata,
            classes_names=("BI", "FI"),
            num_pred=5,
            label_maker=label_maker,
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=5, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
            preprocessing=preprocessing,
        ),
        EvaluateIntermediate(
            data=ds[# To evaluate the correct validation dataset if the model is not trained on the full dataset
                "tr" if config["training_dataset"] == "full"  
                else "tr_"+config["training_dataset"]
                ],
            name="train",
            scheduler=Scheduler(
                every_n_batch=100, every_n_epoch=1, batch_epoch_stop=1,len_data=len_datasets["tr"]
            ),
        ),
    ]
    print(tf.config.list_physical_devices("GPU"))
    ## Training
    model.fit(
        tr_dataset,
        epochs=config["num_epochs"],
        validation_data=ds[
            "val" if config["training_dataset"] == "full" 
                    else "val_"+config["training_dataset"]
                    ],
        callbacks=callbacks,
    )
    ## Final evaluation
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
