import librosa
import librosa.core
import librosa.feature
import librosa.display
import numpy as np
import scipy

import yaml
import logging
import os
import glob

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

# Dataset creation and representation
########################################################################
# all files from directory/section in file list and labels from audio file description
########################################################################

"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def file_list_generator_MIMII(target_dir,
                              section_name,
                              dir_name,
                              mode,
                              prefix_normal="normal",
                              prefix_anomaly="anomaly",
                              ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files
    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))

    # development
    if mode:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                          dir_name=dir_name,
                                                                                                          section_name=section_name,
                                                                                                          prefix_normal=prefix_normal,
                                                                                                          ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                          dir_name=dir_name,
                                                                                                          section_name=section_name,
                                                                                                          prefix_normal=prefix_anomaly,
                                                                                                          ext=ext))
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*.{ext}".format(target_dir=target_dir,
                                                                                        dir_name=dir_name,
                                                                                        section_name=section_name,
                                                                                        ext=ext))
        files = sorted(glob.glob(query))
        labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels


# c Toyadmos
########################################################################
# all files from directory/section in file list and labels from audio file description
########################################################################
def file_list_generator_toyadm(target_dir,
                               dir_name,
                               normal=True,
                               ext="mp4"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files
    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir))

    # development
    # if mode:
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir,
                                                                     dir_name=dir_name,
                                                                     ext=ext))
    normal_files = sorted(glob.glob(query))
    if normal:
        normal_labels = np.zeros(len(normal_files))
    else:
        normal_labels = np.ones(len(normal_files))

    logger.info("#files : {num}".format(num=len(normal_files)))
    if len(normal_files) == 0:
        logger.exception("no_wav_file!!")
    print("\n========================================")

    return normal_files, normal_labels


########################################################################
# feature extractor
########################################################################
def feature_extraction_from_file(file_name, extraction_type='melspectrogram', n_mfcc=40):
    """
    feature extraction for each file

    file_name : str
        target .wav file
    extraction_type : str
        'aggregate_MFCC' - by default
        'amplitude' - original signal, amplitude values timeseries
        'melspectrogram' - 2D melspectrogramm

    return : numpy.array( numpy.array( float ) )
        vector array
        * feature.shape = (1, feature_vector_length)
    """
    y, sr = librosa.load(file_name, sr=16000, mono=True)

    if extraction_type == 'aggregate_MFCC':

        # Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # MFCCs statistics
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.mean(mfccs.T, axis=0)
        mfccs_max = np.mean(mfccs.T, axis=0)
        mfccs_min = np.mean(mfccs.T, axis=0)

        # spectral centroid and statistic
        # Спектральный центроид указывает, на какой частоте сосредоточена энергия спектра (энергия напряжения)
        # т.е указывает, где расположен “центр масс” для звука.
        cent_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                            axis=0)[0]
        cent_std = np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                          axis=0)[0]
        cent_max = np.max(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                          axis=0)[0]
        cent_min = np.min(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                          axis=0)[0]

        features = np.concatenate((
            mfccs_mean, mfccs_std, mfccs_max, mfccs_min), axis=0)

        cent_skew = scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                     axis=0)[0]

        # center frequency for a spectrogram bin such that at least roll_percent (0.85 by default)
        # of the energy of the spectrum in this frame is contained in this bin and the bins below
        rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                               axis=0)[0]
        rolloff_std = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                             axis=0)[0]
        rolloff_max = np.max(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                             axis=0)[0]
        rolloff_min = np.min(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                             axis=0)[0]

        features = np.concatenate((mfccs_mean, mfccs_std, mfccs_max, mfccs_min, np.array(
            (cent_skew, rolloff_mean, rolloff_std, rolloff_max, rolloff_min))), axis=0)
        features = features.reshape(1, -1)

    elif extraction_type == 'amplitude':
        features = y.reshape(1, -1)

    elif extraction_type == 'melspectrogram':
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features = features.reshape(-1, features.shape[0], features.shape[1])
    return features


########################################################################
# all dataset from file_list with features
########################################################################
def file_list_to_data(file_list, extraction_type='melspectrogram', n_mfcc=40):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    extraction_type : str
        'aggregate_MFCC' - by default
        'amplitude' - original signal, amplitude values timeseries
        'melspectrogram' - 2D melspectrogramm
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    if extraction_type == 'melspectrogram':
       
        # iterate file_to_vector_array()
        for idx in tqdm(range(len(file_list))):
           
            vectors = feature_extraction_from_file(
                file_list[idx], extraction_type, n_mfcc)
            vectors = vectors[:: 1, :]
            if idx == 0:
                data = np.zeros(
                    (len(file_list) * vectors.shape[0], 1, n_mfcc, vectors.shape[-1]), float)
            data[vectors.shape[0] * idx: vectors.shape[0]
                 * (idx + 1), :] = vectors

    else:
        # iterate file_to_vector_array()
        for idx in tqdm(range(len(file_list))):
            vectors = feature_extraction_from_file(
                file_list[idx], extraction_type, n_mfcc)
            if idx == 0:
                data = np.zeros(
                    (len(file_list) * vectors.shape[0], vectors.shape[1]), float)
            data[vectors.shape[0] * idx: vectors.shape[0]
                 * (idx + 1), :] = vectors

    return data


# stratified dataset split
def StratifiedKF(X, y):
    """
    X - data
    y - labels
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = []
    for idx in skf.split(X, y):
        idx_list.append(idx)
        train_idx, test_idx = idx_list[0]
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def mix_data(X_list: list, y_list: list):
    """
    mixed data with stratified split

    X_list : list of different parts of data
    y_list: list of different parts of data labels

    return : train, test and validation samples
    """
    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)
    # отделим Val для итоговой классификации по outlier score
    X, X_val, y, y_val = StratifiedKF(X_all, y_all)
    # отделим еще test для валидации модели
    X_train, X_test, y_train, y_test = StratifiedKF(X, y)

    return X_train, X_test, X_val, y_train, y_test, y_val


# normalize data
def scaling(X_train, X_test, X_val=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val