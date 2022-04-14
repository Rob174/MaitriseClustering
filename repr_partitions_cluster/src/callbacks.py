import numpy as np
from pathlib import Path
from h5py import File
from bs4 import BeautifulSoup
import wandb
import tensorflow as tf
import htmlmin
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

class ConfusionMatrix(tf.keras.callbacks.Callback):
    def __init__(self,classes_names:tuple,validation_data,name:str) -> None:
        self.classes_names = classes_names
        self.validation_data = validation_data
        self.name = name
    def on_train_end(self,*args,**kwargs):
        y_true = []
        y_pred = []
        for b_inp,b_y_true in self.validation_data:
            b_y_pred = self.model.predict(b_inp)
            y_true.append(b_y_true)
            y_pred.append(b_y_pred)
        y_true = np.concatenate(y_true,axis=0)
        y_pred = np.concatenate(y_pred,axis=0)
        self.make_confusion_matrix(y_true,y_pred)
    def make_confusion_matrix(self, y_true, y_pred, decimals=2):
        """
        Makes a confusion matrix from the true and predicted labels.
        """
        # Build the confusion matrix
        n = len(self.classes_names)
        confusion_matrix = np.zeros((n, n),dtype=int)
        for i in range(len(y_pred)):
            # Update the confusion matrix
            true_val = np.argmax(y_true[i])
            pred_val = np.argmax(y_pred[i])
            confusion_matrix[pred_val,true_val] += 1
        # Compute additional confusion matrix totals...
        confusion_matrix_percent = confusion_matrix / confusion_matrix.sum() * 100
        tot_row = confusion_matrix.sum(axis=1)
        tot_row_percent = confusion_matrix_percent.sum(axis=1)
        tot_col = confusion_matrix.sum(axis=0)
        tot_col_percent = confusion_matrix_percent.sum(axis=0)
        row_percent = np.divide(confusion_matrix.T, tot_row,where=tot_row!=0).T * 100
        col_percent = np.divide(confusion_matrix, tot_col,where=tot_col!=0) * 100
        accuracy = [np.trace(confusion_matrix)]
        accuracy.append((accuracy[0] / confusion_matrix.sum()*100).round(decimals=decimals))
        # get the template and prepare to load data
        with open(str(Path(".") / "data" / "template_confusion_mat.html"))  as fp:
            content = fp.read()
            soup = BeautifulSoup(content, 'html.parser')
        script_text = [l.strip() for l in soup.findAll("script")[-1].text.split("\n")]
        # Put the data into the template
        base_line = 2
        script_text[2] = "const data = "+str(confusion_matrix.tolist())+";"
        script_text[3] = "const data_percent = "+str(confusion_matrix_percent.round(decimals=decimals).tolist())+";"
        script_text[4] = "const row_percent = "+str(row_percent.round(decimals=decimals).tolist())+";"
        script_text[5] = "const col_percent = "+str(col_percent.round(decimals=decimals).tolist())+";"
        script_text[6] = "const tot_row = "+str(tot_row.tolist())+";"
        script_text[7] = "const tot_row_percent = "+str(tot_row_percent.round(decimals=decimals).tolist())+";"
        script_text[8] = "const tot_col = "+str(tot_col.tolist())+";"
        script_text[9] = "const tot_col_percent = "+str(tot_col_percent.round(decimals=decimals).tolist())+";"
        script_text[10] = "const col_names = "+str(list(self.classes_names))+";"
        script_text[11] = "const accuracy = "+str(accuracy)+";"
        soup.findAll("script")[-1].string = "\n".join(script_text)
        result = str(htmlmin.minify(str(soup), remove_empty_space=True))
        file = str((Path(".") / "data" / "confusion_matrix.html").resolve())
        with open(file, "w") as fp:
            fp.write(result)
        driver = webdriver.Chrome(ChromeDriverManager().install())
        driver.get('file:///'+file)
        html = driver.page_source
        driver.quit()
        # Save to wandb
        html = wandb.Html(html)
        wandb.log({self.name+"_confusion_matrix":html})
        return confusion_matrix

class Predictions(tf.keras.callbacks.Callback):
    def __init__(self,classes_names:tuple,validation_path:Path,path_metadata: Path,num_pred:int=5) -> None:
        self.classes_names = classes_names
        self.num_pred = num_pred
        self.metadata = {}
        with File(str(path_metadata.resolve()), "r") as cache:
            for k,v in cache["metadata"].items():
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
        self.current_batch = 0
        self.current_epoch = 0
        self.pred_every = 100
        self.counter = 0
    def on_train_batch_end(self,batch,logs=None):
        self.current_batch = batch
        self.counter += 1
        if self.counter % self.pred_every == 0:
            self.predict_batch()
    def on_epoch_end(self,epoch,logs=None):
        self.current_epoch = epoch
    def on_train_end(self,*args,**kwargs):
        self.predict_batch()
        
    def predict(self,*args,**kwargs):
        tot_length = 0
        with File(str(self.validation_path.resolve()), "r") as cache:
            keys = list(cache["input"].keys())[:self.num_pred]
            buffer = {}
            for k in keys:
                buffer[k] = {
                    "input": np.copy(cache["input"][k]),
                    "output": np.copy(cache["output"][k]),
                    "metadata": self.metadata[k]
                }
        inputs = np.stack([buffer[k]["input"] for k in keys],axis=0)
        y_pred = self.model.predict(inputs)
        y_true = []
        for k in keys:
            arr = np.zeros((len(self.classes_names),))
            arr[int(buffer[k]["output"][0])] = 1
            y_true.append(arr)
        y_true = np.stack(y_true ,axis=0)
        metadata = [buffer[k]["metadata"]["INIT_CHOICE"] for k in keys]
        keys = list(buffer.keys())
        data = []
        for i,(k,inp,out,pred,metad) in enumerate(zip(keys,inputs,y_true,y_pred,metadata)):
            inp = np.stack([inp[:,:,0],inp[:,:,1],np.zeros(inp[:,:,0].shape)],axis=-1)
            inp = (inp / np.max(inp) * 255).astype(np.uint8)
            img = wandb.Image(inp)
            true = str(self.classes_names[np.argmax(out)]+":"+str(out.round(decimals=2)))
            pred = str(self.classes_names[np.argmax(pred)]+":"+str(pred.round(decimals=2)))
            data.append([self.current_batch,self.current_epoch,k,img,pred,true,metad])
        table = wandb.Table(columns=['batch','epoch','id_dataset','Clustering initial', 'Valeur réellen','Prédiction','INIT_CHOICE'], data=data)
        wandb.log({"predictions":table})
    