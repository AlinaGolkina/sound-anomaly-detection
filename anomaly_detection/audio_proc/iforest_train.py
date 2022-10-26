from sklearn.ensemble import IsolationForest
from dataset import Mic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def iforest_to_onnx(target_dir = r"sound_rec", dir_name_data = r"/train_unlabled"):
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
    clf=IsolationForest(contamination=0.1)
    clf.fit(dataset_train.data)

    # Convert into ONNX format
    initial_type = [('float_input', FloatTensorType([None, dataset_train.data.shape[1]]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    with open("train_iforest.onnx", "wb") as f:
        f.write(onx.SerializeToString())

if __name__ == "__main__":
    iforest_to_onnx()