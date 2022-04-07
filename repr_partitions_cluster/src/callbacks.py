import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
import wandb
import tensorflow as tf
import htmlmin
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

class ConfusionMatrix(tf.keras.callbacks.Callback):
    def __init__(self,classes_names:tuple,validation_data) -> None:
        self.classes_names = classes_names
        self.validation_data = validation_data
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
        wandb.log({"confusion_matrix":html})
        return confusion_matrix

# if __name__ == "__main__":
#     conf = ConfusionMatrix(("0","1"))
#     y_true = np.random.rand(150,2)
#     y_pred = np.random.rand(150,2)
#     conf.make_confusion_matrix(y_true, y_pred)