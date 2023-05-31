import librosa
import numpy as np
import scipy
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path


class IceDataset(Dataset):
    """Parameters:
    -----------
    target_dir : str
        base directory path
    dir_name_normal : str
        sub directory name with normal samples
    dir_name_anomaly : str
        sub directory name with anomaly samples
    extraction_type : str
        'amplitude' - 1D [default] original signal, amplitude values timeseries
        'aggregate_features' - 1D aggregate statistics from melspectrogram, time domain and frequency domain features
        'spectrogram' - 2D spectrogram
        'melspectrogram' - 2D melspectrogramm
        'mfccs' - 2D MFCC coefficients  
        'mfccs_delta' - 2D MFCC coefficients + delta + delta2 mfccs
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

    
    """

    def __init__(
        self,
        target_dir: str,
        dir_name_normal: str,
        dir_name_anomaly,
        extraction_type: str = "amplitude",
        sr: int = 22050,
        n_mfcc: int = 28,
        n_mel: int = 224,
    ):

        self.target_dir = target_dir
        self.dir_name_normal = dir_name_normal
        self.dir_name_anomaly = dir_name_anomaly
        self.extraction_type = extraction_type
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mel = n_mel

        self.samples = []
        self.file_list, self.labels = self._init_file_list_generator()
        self.data = []
        self._file_list_to_data()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx) -> tuple:
        return self.data[idx], self.labels[idx]

    def _init_file_list_generator(self):
        query_norm = Path(
            f"{self.target_dir}/{self.dir_name_normal}"
        ).absolute()
        normal_files = sorted(list(query_norm.glob('*.flac')))
        normal_labels = np.zeros(len(normal_files))

        if self.dir_name_anomaly:
            query_anomaly = Path(
                f"{self.target_dir}/{self.dir_name_anomaly}"
            ).absolute()
            anomaly_files = sorted(list(query_anomaly.glob('*.flac')))
            anomaly_labels = np.ones(len(anomaly_files))

            file_list = np.concatenate((normal_files, anomaly_files), axis=0)
            labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        else:
            file_list = normal_files
            labels = normal_labels

        return file_list, labels

    @staticmethod
    def _feature_extraction_from_file(file, extraction_type, sr, n_mfcc, n_mel):
        """
        feature extractor
        """
        y, sr = librosa.load(file, sr=sr, mono=True)

        if extraction_type == "aggregate_features":

            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # MFCCs statistics
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.mean(mfccs.T, axis=0)
            mfccs_max = np.mean(mfccs.T, axis=0)
            mfccs_min = np.mean(mfccs.T, axis=0)

            rms_mean = np.mean(librosa.feature.rms(y=y).T, axis=0)[0]
            rms_std = np.std(librosa.feature.rms(y=y).T, axis=0)[0]
            rms_max = np.max(librosa.feature.rms(y=y).T, axis=0)[0]
            rms_min = np.min(librosa.feature.rms(y=y).T, axis=0)[0]
            zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]
            zcr_std = np.std(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]
            zcr_max = np.max(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]
            zcr_min = np.min(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]


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
                            rms_mean,
                            rms_std,
                            rms_max,
                            rms_min,
                            zcr_mean,
                            zcr_std,
                            zcr_max,
                            zcr_min,
                        )
                    ),
                ),
                axis=0,
            )
            features = features.reshape(1, -1)

        elif extraction_type == "amplitude":
            features = y.reshape(1, -1)

        elif extraction_type == "mfccs":
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=8900)
            features = features.reshape(-1, features.shape[0], features.shape[1])

        elif extraction_type == "mfccs_delta":
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
            features = features.reshape(-1, features.shape[0], features.shape[1])

        elif extraction_type == "melspectrogram":
            features = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=2048, #1024,
                n_mels=n_mel,
                #win_length=1024,
                hop_length=1084,#512,
                power=2.0,
            )
            features = librosa.power_to_db(features, ref=np.max)
            features = features.reshape(-1, features.shape[0], features.shape[1])

        elif extraction_type == "melspectrogram_aug":
            features = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=1024, #1024,
                n_mels=n_mel,
                #win_length=1024,
                hop_length=512,#512,
                power=2.0,
            )
            features = librosa.power_to_db(features, ref=np.max)
            features = features.reshape(-1, features.shape[0], features.shape[1])

        elif extraction_type == 'spectrogram':
            X = librosa.stft(y)
            features = librosa.amplitude_to_db(abs(X), ref=np.max)
            features = features.reshape(-1, features.shape[0], features.shape[1])

        return features

    def _file_list_to_data(self):
        """
        all dataset from file_list with features
        iteration each file to vector array()
        """
        if self.extraction_type == "melspectrogram" or self.extraction_type == 'spectrogram' or self.extraction_type == 'melspectrogram_aug' or self.extraction_type == 'mfccs' or self.extraction_type == 'mfccs_delta':

            for idx in tqdm(range(len(self.file_list))):

                vectors = IceDataset._feature_extraction_from_file(
                    self.file_list[idx], self.extraction_type, self.sr, self.n_mfcc, self.n_mel
                )
                vectors = vectors[::1, :]
                n_objs = vectors.shape[0]
                if idx == 0:
                    self.data = np.zeros(
                        (
                            len(self.file_list) * n_objs,
                            1,
                            vectors.shape[1],
                            vectors.shape[-1],
                        ),
                        float,
                    )
                self.data[n_objs * idx : n_objs * (idx + 1), :] = vectors

        else:
            for idx in tqdm(range(len(self.file_list))):
                vectors = IceDataset._feature_extraction_from_file(
                    self.file_list[idx], self.extraction_type, self.sr, self.n_mfcc, self.n_mel
                )
                n_objs = vectors.shape[0]
                if idx == 0:
                    self.data = np.zeros(
                        (len(self.file_list) * n_objs, vectors.shape[1]), float
                    )
                self.data[n_objs * idx : n_objs * (idx + 1), :] = vectors

        return self.data
