import os
import shutil

import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import IsolationForest

from .dataset import Mic


def iforest_to_onnx(target_dir=r"sound_rec", dir_name_data=r"/train_unlabled"):
    """
    Anomaly detection training with Isolation Forest
    and saving trained model to onnx

    parameters:
    ----------
    target_dir: str - sound dir
    dir_name_data: str - dir with train data

    return:
    -------
    write trained iForest model to train_iforest.onnx file

    """
    # data dir
    dataset_train = Mic(
        target_dir,
        dir_name_data,
        extraction_type="aggregate_MFCC",
    )
    clf = IsolationForest(contamination=0.1)
    clf.fit(dataset_train.data)

    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, dataset_train.data.shape[1]]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    with open("train_iforest.onnx", "wb") as f:
        f.write(onx.SerializeToString())


def iforest_preds(target_dir, dir_name_data, predicted_dir, onnx_file, batch):
    """
    Get predictions for recordered sounds and move files to predicted_dir
    """
    test = Mic(target_dir, dir_name_data, extraction_type="aggregate_MFCC", batch=batch)

    sess = rt.InferenceSession("train_iforest.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: test.data.astype(np.float32)})[0]
    file_list = [i.split("\\")[-1] for i in test.file_list]
    pred_onx = np.where(pred_onx == -1, "anomaly", "normal")
    file_source = f"{target_dir}/{dir_name_data}"
    file_target = f"{target_dir}/{predicted_dir}"
    for fname, pred in zip(file_list, pred_onx):
        with open("predictions.csv", "a") as file:
            file.write(",".join([str(fname), str(pred)]))
            file.write("\n")

    for file_name in file_list:
        shutil.move(os.path.join(file_source, file_name), file_target)

    return pred_onx


if __name__ == "__main__":
    iforest_preds(
        target_dir=r"sound_rec",
        dir_name_data=r"/record_buffer",
        predicted_dir=r"predicted_records",
        onnx_file="train_iforest.onnx",
        batch=5,
    )
