"""Contains all the callbacks used to evaluate a model takinginto  as input a top view of the initial clustering 
and predicting the best allgorithm or the difference of final cost betweefirst et vebest improvement"""
import numpy as np
from pathlib import Path
from h5py import File
from bs4 import BeautifulSoup
import wandb
import tensorflow as tf
import htmlmin
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


class LabelMaker:
    def get_best(self, value):
        """Get the index of the best class (0=BI, 1=FI)

        Args:
            value (array of np.float32 between 0 and 1): (Pr(BI)  Pr(FI))

        Returns:
            [int]: index of the best class (0=BI, 1=FI)
        """
        return np.argmax(value)

    def make(self, y):
        """Transform a prediction of the network into a probability distribution
        (one for overload purposes here)

        Args:
            y (array of float32 between 0 and 1): (Pr(BI)  Pr(FI)) shape (batch_size, 2)

        Returns:
            (array of float32 between 0 and 1): (Pr(BI)  Pr(FI))  shape (batch_size, 2)
        """
        return y


class OneHotLabelMaker(LabelMaker):
    def get_best(self, value):
        """Get the index of the best class (0=BI, 1=FI)

        Args:
            value (float): prediction of the network in regression mode

        Returns:
            [int]: index of the best class (0=BI, 1=FI)
        """
        if value < 0:
            return 0
        elif value == 0:
            return 0
        else:
            return 1

    def make(self, y):
        """Transform a prediction of the network into a probability distribution
        (one for overload purposes here)

        Args:
            y (ndarray float32): prediction of the network in regression modeshape (batch_size, ) 

        Returns:
            (array of np.float32 between 0 and 1): (Pr(BI)  Pr(FI)) shape (batch_size, 2)
        """
        y = y.flatten()
        labels = np.ones((len(y), 2))
        labels[y > 0, 1] = 0.0
        labels[y < 0, 0] = 0.0
        labels[y == 0, :] = labels[y == 0, :] * 0.5
        return labels

class Scheduler:
    """Schedule callbacks for specific times during training"""
    def __init__(self,len_data: int,every_n_batch: int = None, every_n_epoch: int = None, batch_epoch_stop: int = None):
        """
        Args:
            y (ndarray float32): prediction of the network in regression modeshape (batch_size, ) """
        self.every_n_batch = every_n_batch
        self.every_n_epoch = every_n_epoch
        self.batch_epoch_stop = batch_epoch_stop
        self.current_batch = 0
        self.current_epoch = 0
        self.num_batches_per_epoch = len_data
    def batch_check(self):
        perform_operation = False
        if self.every_n_batch is not None:
            condition = (self.current_batch % self.every_n_batch == 0)
            if self.batch_epoch_stop is not None:
                condition = condition and (self.current_epoch < self.batch_epoch_stop)
            perform_operation = perform_operation or condition
        return perform_operation
    def new_epoch_end(self):
        self.current_epoch += 1
        self.current_batch = 0
        perform_operation = False
        if self.every_n_epoch is not None:
            perform_operation = perform_operation or (self.current_epoch % self.every_n_epoch == 0)
        perform_operation = perform_operation or self.batch_check()
        return perform_operation
    def new_batch_end(self):
        self.current_batch += 1
        return self.batch_check()
    def new_batch_begin(self):
        return self.batch_check()

class ConfusionMatrix(tf.keras.callbacks.Callback):
    """Callback chargée de réaliser une matrice de confusion avec le dataset fourni à interval régulieronn spécifié par schuduler"""
    def __init__(
        self, classes_names: tuple, data, name: str, label_maker: LabelMaker, scheduler: Scheduler
    ) -> None:
        self.classes_names = classes_names
        self.data = data
        self.name = name
        self.label_maker = label_maker
        self.scheduler = scheduler
    def on_epoch_end(self, epoch, logs=None):
        if self.scheduler.new_epoch_end():
            self.make_confusion_matrix(
                name_out=f"{self.name}_confusion_matrix_batch{self.scheduler.current_batch}_epoch{self.scheduler.current_epoch}"
                )
    def on_batch_end(self, batch, logs=None):
        if self.scheduler.new_batch_end():
            self.make_confusion_matrix(
                name_out=f"{self.name}_confusion_matrix_batch{self.scheduler.current_batch}_epoch{self.scheduler.current_epoch}"
                )
    def on_train_end(self, *args, **kwargs):
        self.make_confusion_matrix(
            name_out=f"{self.name}_confusion_matrix_end_training"
            )
    def on_train_begin(self, *args, **kwargs):
        self.make_confusion_matrix(
            name_out=f"{self.name}_confusion_matrix_begin_training"
            )
    def make_confusion_matrix(self, name_out:str, decimals=2):
        """
        Makes a confusion matrix from the true and predicted labels.
        """
        y_true = []
        y_pred = []
        for b_inp, b_y_true in self.data:
            b_y_pred = self.model.predict(b_inp)
            y_true.append(b_y_true)
            y_pred.append(b_y_pred)
        assert len(y_true) > 0, "Must be non-empty"
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = self.label_maker.make(y_true)
        y_pred = self.label_maker.make(y_pred)
        # Build the confusion matrix
        n = len(self.classes_names)
        confusion_matrix = np.zeros((n, n), dtype=int)
        for i in range(len(y_pred)):
            # Update the confusion matrix
            true_val = np.argmax(y_true[i])
            pred_val = np.argmax(y_pred[i])
            confusion_matrix[pred_val, true_val] += 1
        # Compute additional confusion matrix totals...
        confusion_matrix_percent = confusion_matrix / confusion_matrix.sum() * 100
        tot_row = confusion_matrix.sum(axis=1)
        tot_row_percent = confusion_matrix_percent.sum(axis=1)
        tot_col = confusion_matrix.sum(axis=0)
        tot_col_percent = confusion_matrix_percent.sum(axis=0)
        row_percent = np.divide(confusion_matrix.T, tot_row, where=tot_row != 0).T * 100
        col_percent = np.divide(confusion_matrix, tot_col, where=tot_col != 0) * 100
        accuracy = [np.trace(confusion_matrix)]
        accuracy.append(
            (accuracy[0] / confusion_matrix.sum() * 100).round(decimals=decimals)
        )
        # get the template and prepare to load data
        with open(str(Path(".") / "data" / "template_confusion_mat.html")) as fp:
            content = fp.read()
            soup = BeautifulSoup(content, "html.parser")
        script_text = [l.strip() for l in soup.findAll("script")[-1].text.split("\n")]
        # Put the data into the template
        base_line = 2
        script_text[2] = "const data = " + str(confusion_matrix.tolist()) + ";"
        script_text[3] = (
            "const data_percent = "
            + str(confusion_matrix_percent.round(decimals=decimals).tolist())
            + ";"
        )
        script_text[4] = (
            "const row_percent = "
            + str(row_percent.round(decimals=decimals).tolist())
            + ";"
        )
        script_text[5] = (
            "const col_percent = "
            + str(col_percent.round(decimals=decimals).tolist())
            + ";"
        )
        script_text[6] = "const tot_row = " + str(tot_row.tolist()) + ";"
        script_text[7] = (
            "const tot_row_percent = "
            + str(tot_row_percent.round(decimals=decimals).tolist())
            + ";"
        )
        script_text[8] = "const tot_col = " + str(tot_col.tolist()) + ";"
        script_text[9] = (
            "const tot_col_percent = "
            + str(tot_col_percent.round(decimals=decimals).tolist())
            + ";"
        )
        script_text[10] = "const col_names = " + str(list(self.classes_names)) + ";"
        script_text[11] = "const accuracy = " + str(accuracy) + ";"
        soup.findAll("script")[-1].string = "\n".join(script_text)
        result = str(htmlmin.minify(str(soup), remove_empty_space=True))
        file = str((Path(".") / "data" / "confusion_matrix.html").resolve())
        with open(file, "w") as fp:
            fp.write(result)
        driver = webdriver.Chrome(ChromeDriverManager().install())
        driver.get("file:///" + file)
        html = driver.page_source
        driver.quit()
        # Save to wandb
        html = wandb.Html(html)
        wandb.log({name_out: html})
        return confusion_matrix

class EvaluateIntermediate(tf.keras.callbacks.Callback):
    """Compute all metrics as with keras. 
    Difference : computed on the full dataset and can be scheduled in the middle of an epoch thanks to the scheduler"""
    def __init__(self, data: Path, name: str, scheduler: Scheduler) -> None:
        super().__init__()
        self.data = data
        self.name = name
        self.scheduler = scheduler
    def on_train_begin(self, *args, **kwargs):
        self.evaluation()
    def on_batch_end(self, batch, logs=None):
        if self.scheduler.new_batch_end():
            self.evaluation()
    def on_epoch_end(self, epoch, logs=None):
        if self.scheduler.new_epoch_end():
            self.evaluation()
    def evaluation(self):
        Llogs  = []
        size = 0
        for b_inp, b_y_true in self.data:
            Llogs.append(self.model.evaluate(b_inp,b_y_true))
            size +=  len(b_inp)
        logs = np.sum(Llogs,axis=0)*wandb.config["batch_size"]/size
        wandb.log({
            "frac_epoch": self.scheduler.current_epoch+self.scheduler.current_batch/self.scheduler.num_batches_per_epoch ,
            self.name+"_n_batch":self.scheduler.current_batch,
            self.name+"_n_samples_seen":self.scheduler.current_batch*wandb.config["batch_size"],
            self.name+"_loss": logs[0],
            self.name+"_MAE": logs[1],
            self.name+"_accuracy": logs[2]
            })
class Predictions(tf.keras.callbacks.Callback):
    """Make the specified number of predictions on the dataset provided. 
    Can be scheduled in the middle of an epoch thanks to the scheduler"""
    def __init__(
        self,
        name: str,
        classes_names: tuple,
        validation_path: Path,
        path_metadata: Path,
        label_maker: LabelMaker,
        scheduler: Scheduler,
        preprocessing,
        num_pred: int = 5,
    ) -> None:
        self.classes_names = classes_names
        self.num_pred = num_pred
        self.label_maker = label_maker
        self.name = name
        self.scheduler = scheduler
        self.preprocessing = preprocessing
        self.data = []
        self.metadata = {}
        with File(str(path_metadata.resolve()), "r") as cache:
            for k, v in cache["metadata"].items():
                arr = np.copy(v)
                if (arr.shape[0]) == 11:
                    self.metadata[k] = {
                        "SEED": arr[0],
                        "NUM_CLUST": arr[1],
                        "NUM_POINTS": arr[2],
                        "INIT_CHOICE": "random" if arr[3] == 0 else "kmeans+",
                        "IMPR_CLASS": "BI" if arr[4] == 0 else "FI",
                        "IT_ORDER": "BACK" if arr[5] == 0 else "other",
                        "init_cost": arr[6],
                        "final_cost": arr[7],
                        "num_iter": arr[8],
                        "num_iter_glob": arr[9],
                        "duration": arr[10],
                    }
                elif (arr.shape[0]) == 12:
                    self.metadata[k] = {
                        "SEED_POINTS": arr[0],
                        "SEED_ASSIGNS": arr[1],
                        "NUM_CLUST": arr[2],
                        "NUM_POINTS": arr[3],
                        "INIT_CHOICE": "random" if arr[4] == 0 else "kmeans+",
                        "IMPR_CLASS": "BI" if arr[5] == 0 else "FI",
                        "IT_ORDER": "BACK" if arr[6] == 0 else "other",
                        "init_cost": arr[7],
                        "final_cost": arr[8],
                        "num_iter": arr[9],
                        "num_iter_glob": arr[10],
                        "duration": arr[11],
                    }
                else:
                    raise Exception("Metadata file has wrong shape")

        self.validation_path = validation_path
        self.id = 0

    def on_train_begin(self, batch, logs=None):
        self.predict()

    def on_train_batch_end(self, batch, logs=None):
        if self.scheduler.new_batch_end():
            self.predict()

    def on_epoch_end(self, epoch, logs=None):
        if self.scheduler.new_epoch_end():
            self.predict()

    def on_train_end(self, *args, **kwargs):
        self.predict()

    def predict(self, *args, **kwargs):
        with File(str(self.validation_path.resolve()), "r") as cache:
            keys = list(cache["input"].keys())
            buffer = {}
            categories_seen = {"BI":0,"FI":0}
            for k in keys:
                out = np.copy(cache["output"][k])
                best = self.classes_names[self.label_maker.get_best(out)]
                tmp_cpy_categories_seen = {k:v for k,v in categories_seen.items()}
                tmp_cpy_categories_seen[best] += 1
                if abs(tmp_cpy_categories_seen["BI"]-tmp_cpy_categories_seen["FI"]) <= 1:
                    categories_seen[best] += 1
                    buffer[k] = {
                        "input": np.copy(cache["input"][k]),
                        "output": out,
                        "metadata": self.metadata[k],
                    }
                if len(buffer) == self.num_pred:
                    break
        inputs_orig = np.stack([buffer[k]["input"] for k in buffer.keys()], axis=0)
        inputs = self.preprocessing(inputs_orig).numpy()
        statistics = {}
        for i,k in enumerate(buffer):
            inp = inputs[i]
            statistics[k] = {"min": np.min(inp), "max": np.max(inp),"mean": np.mean(inp),"std": np.std(inp)}
            if statistics[k]["max"] > 1.0:
                raise Exception
        y_true = []
        for k in buffer:
            y_true.append(buffer[k]["output"])
            
        y_pred = self.model.predict(inputs)
        
        metrics = []
        for inp,true in zip(inputs,y_true):
            metrics.append(self.model.evaluate(np.reshape(inp,(1,*inp.shape)),np.reshape(true,(1,*true.shape))))
        y_true = np.stack(y_true, axis=0)
        metadata = [buffer[k]["metadata"]["INIT_CHOICE"] for k in buffer]
        keys = list(buffer.keys())
        for k, inp, out, pred, metad,metr in zip(keys, inputs, y_true, y_pred, metadata, metrics):
            inp = np.stack(
                [inp[:, :, 0], inp[:, :, 1], np.zeros(inp[:, :, 0].shape)], axis=-1
            )
            inp = (inp / np.max(inp) * 255).astype(np.uint8)
            img = wandb.Image(inp)
            true = str(
                self.classes_names[self.label_maker.get_best(out)]
                + ":"
                + str(out.round(decimals=6))
            )
            pred = str(
                "Pred:"
                + self.classes_names[self.label_maker.get_best(pred)]
                + ":"
                + str(pred.round(decimals=6))
            )
            self.data.append(
                [
                    self.id,
                    self.scheduler.current_batch,
                    self.scheduler.current_epoch,
                    k,
                    img,
                    true,
                    pred,
                    metad,
                    metr[0],
                    metr[1],
                    metr[2],
                    *[stat for stat in statistics[k].values()] 
                ]
            )
            self.id += 1
        table = wandb.Table(
            columns=[
                "ID",
                "batch",
                "epoch",
                "id_dataset",
                "Clustering initial",
                "Valeur réelles",
                "Prédiction",
                "INIT_CHOICE",
                "loss",
                "MAE",
                "accuracy",
                *[f"input_"+stat_name for stat_name in statistics[list(statistics.keys())[0]].keys()],
            ],
            data=self.data,
        )
        wandb.log(
            {
                "predictions_"+self.name: table,
                "epoch": self.scheduler.current_epoch,
                "batch": self.scheduler.current_batch,
            }
        )
