import onnxruntime as rt
import numpy as np
from dataset import Mic
import shutil
import os

def iforest_preds(target_dir, dir_name_data, predicted_dir, onnx_file, batch):
    """
    Get predictions for recordered sounds and move files to predicted_dir
    """
    test = Mic(
        target_dir,
        dir_name_data,
        extraction_type="aggregate_MFCC",
        batch=batch
    )
    
    sess = rt.InferenceSession("train_iforest.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: test.data.astype(np.float32)})[0]
    file_list = [i.split('\\')[-1] for i in test.file_list]
    pred_onx = np.where(pred_onx==-1,'anomaly','normal')
    file_source = f'{target_dir}/{dir_name_data}'
    file_target = f'{target_dir}/{predicted_dir}'
    for i, j in zip(file_list, pred_onx):
        with open('predictions.csv', 'a') as fd:
            fd.write(','.join([str(i),str(j)]))
            fd.write('\n')
    
    for file_name in file_list:
        shutil.move(os.path.join(file_source, file_name), file_target)
    return pred_onx

if __name__ == "__main__":
    iforest_preds(target_dir = r"sound_rec", dir_name_data=r"/record_buffer", predicted_dir=r"predicted_records", onnx_file='train_iforest.onnx', batch = 5)