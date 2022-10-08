import glob
import os

import librosa
import numpy as np
import scipy
from torch.utils.data import Dataset
from tqdm import tqdm


class MimiiDue(Dataset):
    """Parameters:
    -----------
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name (train or source_test or target_test)
    mode : str
        'development' - [default] (with leabels normal/anomal)
        'evaluation' - audio without lables
    extraction_type : str
        'amplitude' - [default] original signal, amplitude values timeseries
        'aggregate_MFCC' - aggregate statistic from melspectrogram
        'melspectrogram' - 2D melspectrogramm
    n_mfcc : int
        mel koeff q-ty
        40 - [default]

    return:
    --------
        if the mode is "development":
            data: audio_files_features according extraction_type: numpy.array (numpy.array( float )) vector array
            labels : np.array [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
             data: audio_files_features, np.array (numpy.array( float )) vector array

    dataset download:
    -----------------
        Download dev_data_<machine_type>.zip from https://zenodo.org/record/4562016
    directory structure:
      /dev_data
        /gearbox
            /train (Normal data in the source and target domains for all sections are included.)
                /section_00_source_train_normal_0000_.wav
                /section_00_source_train_normal_0001_.wav
                ...
                /section_00_target_train_normal_0000_.wav
                /section_00_target_train_normal_0001_.wav
                ...
            /source_test (Normal and anomaly data in the source domain for all sections are included.)
                /section_00_source_test_normal_0099.wav
                /section_00_source_test_anomaly_0000.wav
                ...
                /section_00_source_test_anomaly_0099.wav
                /section_01_source_test_normal_0000.wav
            /target_test (Normal and anomaly data in the target domain for all sections are included.)
                /section_00_target_test_normal_0099.wav
                /section_00_target_test_anomaly_0000.wav
                ...
                /section_00_target_test_anomaly_0099.wav
                /section_01_target_test_normal_0000.wav
                ...
    """

    def __init__(
        self,
        target_dir: str,
        section_name: str,
        dir_name: str,
        mode: str = "development",
        extraction_type: str = "amplitude",
        n_mfcc: int = 40,
    ):
        self.target_dir = target_dir
        self.section_name = section_name
        self.dir_name = dir_name
        self.mode = mode
        self.extraction_type = extraction_type
        self.n_mfcc = n_mfcc

        self.samples = []
        self.file_list, self.labels = self._init_file_list_generator()
        self.data = []
        self._file_list_to_data()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx], self.labels[idx]

    def _init_file_list_generator(self):
        if self.mode:
            # development
            query = os.path.abspath(
                f"{self.target_dir}/{self.dir_name}/{self.section_name}_*_normal_*.wav"
            )
            normal_files = sorted(glob.glob(query))
            normal_labels = np.zeros(len(normal_files))

            query = os.path.abspath(
                "{target_dir}/{dir_name}/{section_name}_*_anomaly_*.wav".format(
                    target_dir=self.target_dir,
                    dir_name=self.dir_name,
                    section_name=self.section_name,
                )
            )
            anomaly_files = sorted(glob.glob(query))
            anomaly_labels = np.ones(len(anomaly_files))

            file_list = np.concatenate((normal_files, anomaly_files), axis=0)
            labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        else:
            # evaluation
            query = os.path.abspath(
                "{target_dir}/{dir_name}/{section_name}_*.wav".format(
                    target_dir=self.target_dir,
                    dir_name=self.dir_name,
                    section_name=self.section_name,
                )
            )
            file_list = [sorted(glob.glob(query))]
            labels = [None]

        return file_list, labels

    @staticmethod
    def _feature_extraction_from_file(file, extraction_type, n_mfcc):
        """
        feature extractor
        """
        y, sr = librosa.load(file, sr=16000, mono=True)

        if extraction_type == "aggregate_MFCC":

            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # MFCCs statistics
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.mean(mfccs.T, axis=0)
            mfccs_max = np.mean(mfccs.T, axis=0)
            mfccs_min = np.mean(mfccs.T, axis=0)

            # spectral centroid and statistic
            # Спектральный центроид указывает, на какой частоте сосредоточена энергия спектра (энергия напряжения)
            # т.е указывает, где расположен “центр масс” для звука.
            cent_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[
                0
            ]
            cent_std = np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0]
            cent_max = np.max(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0]
            cent_min = np.min(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0]
            cent_skew = scipy.stats.skew(
                librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0
            )[0]

            # center frequency for a spectrogram bin such that at least roll_percent (0.85 by default)
            # of the energy of the spectrum in this frame is contained in this bin and the bins below
            rolloff_mean = np.mean(
                librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0
            )[0]
            rolloff_std = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[
                0
            ]
            rolloff_max = np.max(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[
                0
            ]
            rolloff_min = np.min(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[
                0
            ]

            features = np.concatenate(
                (
                    mfccs_mean,
                    mfccs_std,
                    mfccs_max,
                    mfccs_min,
                    np.array(
                        (
                            cent_mean,
                            cent_skew,
                            rolloff_mean,
                            cent_std,
                            cent_max,
                            cent_min,
                            rolloff_std,
                            rolloff_max,
                            rolloff_min,
                        )
                    ),
                ),
                axis=0,
            )
            features = features.reshape(1, -1)

        elif extraction_type == "amplitude":
            features = y.reshape(1, -1)

        elif extraction_type == "melspectrogram":
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            features = features.reshape(-1, features.shape[0], features.shape[1])
        return features

    def _file_list_to_data(self):
        """all dataset from file_list with features
        iteration each file to vector array()
        """
        if self.extraction_type == "melspectrogram":

            for idx in tqdm(range(len(self.file_list))):

                vectors = MimiiDue._feature_extraction_from_file(
                    self.file_list[idx], self.extraction_type, self.n_mfcc
                )
                vectors = vectors[::1, :]
                n_objs = vectors.shape[0]
                if idx == 0:
                    self.data = np.zeros(
                        (
                            len(self.file_list) * n_objs,
                            1,
                            self.n_mfcc,
                            vectors.shape[-1],
                        ),
                        float,
                    )
                self.data[n_objs * idx : n_objs * (idx + 1), :] = vectors
        else:
            for idx in tqdm(range(len(self.file_list))):
                vectors = MimiiDue._feature_extraction_from_file(
                    self.file_list[idx], self.extraction_type, self.n_mfcc
                )
                n_objs = vectors.shape[0]
                if idx == 0:
                    self.data = np.zeros(
                        (len(self.file_list) * n_objs, vectors.shape[1]), float
                    )
                self.data[n_objs * idx : n_objs * (idx + 1), :] = vectors

        return self.data


class ToyAdmos(Dataset):
    """Parameters:
    -----------
    target_dir : str
        base directory path
    dir_name_normal : str
        sub directory name with normal samples
    dir_name_anomaly : str
        sub directory name with anomaly samples
    extraction_type : str
        'amplitude' - [default] original signal, amplitude values timeseries
        'aggregate_MFCC' - aggregate statistic from melspectrogram
        'melspectrogram' - 2D melspectrogramm
    n_mfcc : int
        mel koeff q-ty
        40 - [default]

    return:
    --------
        full dataset (train + test) audio_files_features according extraction_type:
        data: numpy.array (numpy.array( float )) vector array of features
        labels : np.array [ boolean ]
                 label info. list
                * normal/anomaly = 0/1

    dataset download:
    -----------------
        Download toyad2_car_A_anomaly.zip, toyad2_car_A1_normal, ... from https://zenodo.org/record/4580270#.Yzh9sDPP1D-
        directory structure:
        /ToyAdmos2
            /toyad2_car_A_anomaly
                /CA001-carA1-speed1-a-damageL_mic1_00001.mp4
                /CA001-carA1-speed1-a-damageL_mic1_00002.mp4
                ...
            /toyad2_car_A1_normal
                /CN001-carA1-speed1_mic1_00001.mp4
                /CN001-carA1-speed1_mic1_00002.mp4
                ...

            /toyad2_car_A2_normal
                /CN003-carA2-speed1_mic1_00001.mp4
                /CN003-carA2-speed1_mic1_00002.mp4
                ...
    """

    def __init__(
        self,
        target_dir,
        dir_name_normal,
        dir_name_anomaly,
        extraction_type="amplitude",
        n_mfcc=40,
    ):
        self.extraction_type = extraction_type
        self.target_dir = target_dir
        self.dir_name_normal = dir_name_normal
        self.dir_name_anomaly = dir_name_anomaly
        self.n_mfcc = n_mfcc
        self.samples = []
        self.labels = []
        self.file_list = []
        self.__init_file_list_generator()
        self.data = []
        self._file_list_to_data()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __init_file_list_generator(self):
        query_norm = os.path.abspath(
            "{target_dir}/{dir_name_normal}/*.mp4".format(
                target_dir=self.target_dir, dir_name_normal=self.dir_name_normal
            )
        )
        normal_files = sorted(glob.glob(query_norm))
        normal_labels = np.zeros(len(normal_files))

        query_anomaly = os.path.abspath(
            "{target_dir}/{dir_name_anomaly}/*.mp4".format(
                target_dir=self.target_dir, dir_name_anomaly=self.dir_name_anomaly
            )
        )
        anomaly_files = sorted(glob.glob(query_anomaly))
        anomaly_labels = np.ones(len(anomaly_files))

        self.file_list = np.concatenate((normal_files, anomaly_files), axis=0)
        self.labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        return self.file_list, self.labels

    @staticmethod
    def _feature_extraction_from_file(file, extraction_type, n_mfcc):
        """feature extractor"""
        y, sr = librosa.load(file, sr=16000, mono=True)

        if extraction_type == "aggregate_MFCC":

            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # MFCCs statistics
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.mean(mfccs.T, axis=0)
            mfccs_max = np.mean(mfccs.T, axis=0)
            mfccs_min = np.mean(mfccs.T, axis=0)

            # spectral centroid and statistic
            # Спектральный центроид указывает, на какой частоте сосредоточена энергия спектра (энергия напряжения)
            # т.е указывает, где расположен “центр масс” для звука.
            cent_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[
                0
            ]
            cent_std = np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0]
            cent_max = np.max(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0]
            cent_min = np.min(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0]
            cent_skew = scipy.stats.skew(
                librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0
            )[0]

            # center frequency for a spectrogram bin such that at least roll_percent (0.85 by default)
            # of the energy of the spectrum in this frame is contained in this bin and the bins below
            rolloff_mean = np.mean(
                librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0
            )[0]
            rolloff_std = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[
                0
            ]
            rolloff_max = np.max(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[
                0
            ]
            rolloff_min = np.min(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[
                0
            ]

            features = np.concatenate(
                (
                    mfccs_mean,
                    mfccs_std,
                    mfccs_max,
                    mfccs_min,
                    np.array(
                        (
                            cent_mean,
                            cent_std,
                            cent_max,
                            cent_min,
                            cent_skew,
                            rolloff_mean,
                            rolloff_std,
                            rolloff_max,
                            rolloff_min,
                        )
                    ),
                ),
                axis=0,
            )
            features = features.reshape(1, -1)

        elif extraction_type == "amplitude":
            features = y.reshape(1, -1)

        elif extraction_type == "melspectrogram":
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            features = features.reshape(-1, features.shape[0], features.shape[1])
        return features

    ########################################################################
    # all dataset from file_list with features
    ########################################################################

    def _file_list_to_data(self):
        """iteration each file to vector array()"""
        if self.extraction_type == "melspectrogram":

            for idx in tqdm(range(len(self.file_list))):

                vectors = ToyAdmos._feature_extraction_from_file(
                    self.file_list[idx], self.extraction_type, self.n_mfcc
                )
                vectors = vectors[::1, :]
                if idx == 0:
                    self.data = np.zeros(
                        (
                            len(self.file_list) * vectors.shape[0],
                            1,
                            self.n_mfcc,
                            vectors.shape[-1],
                        ),
                        float,
                    )
                self.data[
                    vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :
                ] = vectors

        else:
            for idx in tqdm(range(len(self.file_list))):
                vectors = ToyAdmos._feature_extraction_from_file(
                    self.file_list[idx], self.extraction_type, self.n_mfcc
                )
                n_objs = vectors.shape[0]
                if idx == 0:
                    self.data = np.zeros(
                        (len(self.file_list) * n_objs, vectors.shape[1]), float
                    )
                self.data[n_objs * idx : n_objs * (idx + 1), :] = vectors

        return self.data
