## 音频驱动视频生成

[源代码下载](../../video_gen.zip)

1. [IP_LAP-main](#ilm)
    
    1. [音频预处理](#app)
    2. [视频预处理](#vpp)
    3. [audio.py](#ap)
    4. [生成landmark](#gl)
    5. [生成pixmap](#gp)
    6. [生成视频](#gv)
    7. [绘制landmark](#dl)
    8. [推理](#is)
    
2. [Wav2lip](#W2l)
    
     	1. [面部检测模型](#fd)
     	2. [conv](#conv)
     	3. [syncnet](#sync)
     	4. [wav2Lip](#w2l)
     	5. [gen_videos_from_filelist](#gvff)
     	6. [real_videos_inference](#rvi)
    
    

#### IP_LAP-main{#ilm}
#### 音频预处理{#app}

```py
import sys
sys.path.append("..")
from  models import audio
from os import  path
from concurrent.futures import as_completed, ProcessPoolExecutor
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--process_num', type=int, default=6) #number of process to preprocess the audio
parser.add_argument("--data_root", type=str,help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--out_root", help="output audio root", required=True)
args = parser.parse_args()
sample_rate=16000  # 16000Hz
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.out_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)
    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile.replace(' ', r'\ '), wavpath.replace(' ', r'\ '))
    subprocess.run(command, shell=True)
    wav = audio.load_wav(wavpath, sample_rate)
    orig_mel = audio.melspectrogram(wav).T
    np.save(path.join(fulldir, 'audio'), orig_mel)


def mp_handler_audio(job):
    vfile, args = job
    try:
        process_audio_file(vfile, args)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print("looking up paths.... from", args.data_root)
    filelist = glob(path.join(args.data_root, '*/*.mp4'))

    jobs = [(vfile, args) for i, vfile in enumerate(filelist)]
    p_audio = ProcessPoolExecutor(args.process_num)
    futures_audio = [p_audio.submit(mp_handler_audio, j) for j in jobs]

    _ = [r.result() for r in tqdm(as_completed(futures_audio), total=len(futures_audio))]
    print("complete, output to",args.out_root)

if __name__ == '__main__':
    main(args)
```

#### 视频预处理{#vpp}

```py
import os.path
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, traceback
from tqdm import tqdm
import glob
import argparse
import math
from typing import List, Mapping, Optional, Tuple, Union
import cv2
import dataclasses
import numpy as np
from mediapipe.framework.formats import landmark_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--process_num', type=int, default=6) #number of process in ThreadPool to preprocess the dataset
parser.add_argument('--dataset_video_root', type=str, required=True)
parser.add_argument('--output_sketch_root', type=str, default='./lrs2_sketch128')
parser.add_argument('--output_face_root', type=str, default='./lrs2_face128')
parser.add_argument('--output_landmark_root', type=str, default='./lrs2_landmarks')

args = parser.parse_args()

input_mp4_root = args.dataset_video_root
output_sketch_root = args.output_sketch_root
output_face_root=args.output_face_root
output_landmark_root=args.output_landmark_root



"""MediaPipe solution drawing utils."""
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), 
                                (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), 
                                (93, 234),
                                (234, 127), (127, 162)])
# (10, 338), (338, 297), (297, 332), (332, 284),(284, 251), (251, 389) (162, 21), (21, 54),(54, 103), (103, 67), (67, 109), (109, 10)

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), \
                           (4, 45), (45, 220), (220, 115), (115, 48), \
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_FULL = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])
def summarize_landmarks(edge_set):
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmark_idx = summarize_landmarks(FACEMESH_FULL)
pose_landmark_idx = \
    summarize_landmarks(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE, \
                                              FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])
content_landmark_idx = all_landmark_idx - pose_landmark_idx

def draw_landmarks(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec,
        Mapping[int, DrawingSpec]] = DrawingSpec(
            color=RED_COLOR),
        connection_drawing_spec: Union[DrawingSpec,
        Mapping[Tuple[int, int],
        DrawingSpec]] = DrawingSpec()):
    """Draws the landmarks and the connections on the image.
  Args:
    image: A three channel BGR image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from
      hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius.
      If this argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from
      hand connections to the DrawingSpecs that specifies the
      connections' drawing settings such as color and line thickness.
      If this argument is explicitly set to None, no landmark connections will
      be drawn.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel BGR.
      b) If any connetions contain invalid landmark index.
  """
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                # if start_idx in content_landmark and end_idx in content_landmark:
                cv2.line(image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def process_video_file(mp4_path):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        video_stream = cv2.VideoCapture(mp4_path)
        fps = round(video_stream.get(cv2.CAP_PROP_FPS))
        if fps != 25:
            print(mp4_path, ' fps is not 25!!!')
            exit()
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)

        for frame_idx,full_frame in enumerate(frames):
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue  # not detect
            face_landmarks=results.multi_face_landmarks[0]

            #(1)normalize landmarks
            x_min=999
            x_max=-999
            y_min=999
            y_max=-999
            pose_landmarks, content_landmarks = [], []
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in all_landmark_idx:
                    if landmark.x<x_min:
                        x_min=landmark.x
                    if landmark.x>x_max:
                        x_max=landmark.x

                    if landmark.y<y_min:
                        y_min=landmark.y
                    if landmark.y>y_max:
                        y_max=landmark.y
                ######
                if idx in pose_landmark_idx:
                    pose_landmarks.append((idx,landmark.x,landmark.y))
                if idx in content_landmark_idx:
                    content_landmarks.append((idx,landmark.x,landmark.y))
            ##########plus 5 pixel to size##########
            x_min=max(x_min-5/w,0)
            x_max = min(x_max + 5 / w, 1)
            #
            y_min = max(y_min - 5 / h, 0)
            y_max = min(y_max + 5 / h, 1)
            face_frame=cv2.resize(full_frame[int(y_min*h):int(y_max*h),int(x_min*w):int(x_max*w)],(128,128))

            # update landmarks
            pose_landmarks=[ \
                (idx,(x-x_min)/(x_max-x_min),(y-y_min)/(y_max-y_min)) for idx,x,y in pose_landmarks]
            content_landmarks=[\
                (idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)) for idx, x, y in content_landmarks]
            # update drawed landmarks
            for idx,x,y in pose_landmarks + content_landmarks:
                face_landmarks.landmark[idx].x=x
                face_landmarks.landmark[idx].y=y
            #save landmarks
            result_dict={}
            result_dict['pose_landmarks']=pose_landmarks
            result_dict['content_landmarks']=content_landmarks
            out_dir = os.path.join(output_landmark_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir,str(frame_idx)),result_dict)

            #save sketch
            h_new=(y_max-y_min)*h
            w_new = (x_max - x_min) * w
            annotated_image = np.zeros((int(h_new * 128 / min(h_new, w_new)), int(w_new * 128 / min(h_new, w_new)), 3))
            draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,  # FACEMESH_CONTOURS  FACEMESH_LIPS
                connections=FACEMESH_FULL,
                connection_drawing_spec=drawing_spec)  # landmark_drawing_spec=None,
            annotated_image = cv2.resize(annotated_image, (128, 128))

            out_dir = os.path.join(output_sketch_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, str(frame_idx)+'.png'), annotated_image)

            #save face frame
            out_dir = os.path.join(output_face_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, str(frame_idx) + '.png'), face_frame)

def mp_handler(mp4_path):
    try:
        process_video_file(mp4_path)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main():
    print('looking up videos.... ')
    mp4_list = glob.glob(input_mp4_root + '/*/*.mp4')  #example: .../lrs2_video/5536038039829982468/00001.mp4
    print('total videos :', len(mp4_list))

    process_num = args.process_num
    print('process_num: ', process_num)
    p_frames = ThreadPoolExecutor(process_num)
    futures_frames = [p_frames.submit(mp_handler, mp4_path) for mp4_path in mp4_list]
    _ = [r.result() for r in tqdm(as_completed(futures_frames), total=len(futures_frames))]
    print("complete task!")

if __name__ == '__main__':
    main()

```

#### audio.py{#ap}

```py
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import lws

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters
hp = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,

    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
    # be too big to avoid gradient explosion,
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.

    ###################### Our training parameters #################################
    img_size=288,
    fps=25,

    batch_size=8,
    initial_learning_rate=1e-4,
    nepochs=200000000000000000,
    ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
    num_workers=4,
    checkpoint_interval=6000,
    eval_interval=6000,
    save_optimizer_state=True,

    syncnet_wt=0.0,  # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence.
    syncnet_batch_size=128,
    syncnet_lr=1e-4,
    syncnet_eval_interval=4500,
    syncnet_checkpoint_interval=4500,

    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)


def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

```

#### 生成landmark{#gl}

```py
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import lws

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters
hp = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,

    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
    # be too big to avoid gradient explosion,
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.

    ###################### Our training parameters #################################
    img_size=288,
    fps=25,

    batch_size=8,
    initial_learning_rate=1e-4,
    nepochs=200000000000000000,
    ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
    num_workers=4,
    checkpoint_interval=6000,
    eval_interval=6000,
    save_optimizer_state=True,

    syncnet_wt=0.0,  # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence.
    syncnet_batch_size=128,
    syncnet_lr=1e-4,
    syncnet_eval_interval=4500,
    syncnet_checkpoint_interval=4500,

    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)


def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

```

#### 生成pixmap{#gp}

```py
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_D(input_nc=3, ndf=64, n_layers_D=3, norm='instance', use_sigmoid=False, num_D=2, getIntermFeat=True):
    #('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    #('--input_nc', type=int, default=3, help='# of input image channels')
    #('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
   # ('--num_D', type=int, default=2, help='number of discriminators to use')

    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    #print(netD)
    netD.apply(weights_init)
    return netD


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):

        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer




class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input): #: (B,T,C,H,W)
        # input = torch.cat([input[i,:] for i in range(input.size(0))], dim=0)# : (B*T,C,H,W)
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
```

#### 生成视频{#gv}

```py
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision



class AdaINLayer(nn.Module):
    def __init__(self, input_nc, modulation_nc):
        super().__init__()

        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):

        # Part 1. generate parameter-free normalized activations
        normalized = self.InstanceNorm2d(input)

        # Part 2. produce scaling and bias conditioned on feature
        modulation_input = modulation_input.view(modulation_input.size(0), -1)#B 512
        actv = self.mlp_shared(modulation_input)# b 128
        gamma = self.mlp_gamma(actv)# b 128
        beta = self.mlp_beta(actv)# 128

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out

class AdaIN(torch.nn.Module):

    def __init__(self, input_channel, modulation_channel,kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel)

    def forward(self, x, modulation):

        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x




class SPADELayer(torch.nn.Module):
    def __init__(self, input_channel, modulation_channel, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADELayer, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(input_channel)

        self.conv1 = torch.nn.Conv2d(modulation_channel, hidden_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        self.gamma = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.beta = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input, modulation):
        norm = self.instance_norm(input)

        conv_out = self.conv1(modulation)

        gamma = self.gamma(conv_out)
        beta = self.beta(conv_out)

        return norm + norm * gamma + beta


class SPADE(torch.nn.Module):
    def __init__(self, num_channel, num_channel_modulation, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADE, self).__init__()
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.spade_layer_1 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)
        self.spade_layer_2 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)

    def forward(self, input, modulations):
        input = self.spade_layer_1(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_1(input)
        input = self.spade_layer_2(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_2(input)
        return input

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

def downsample(x, size):
    if len(x.size()) == 5:
        size = (x.size(2), size[0], size[1])
        return torch.nn.functional.interpolate(x, size=size, mode='nearest')
    return  torch.nn.functional.interpolate(x, size=size, mode='nearest')


def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b, c, h, w = flow.shape
    flow_norm = 2 * torch.cat([flow[:, :1, ...] / (w - 1), flow[:, 1:, ...] / (h - 1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0, 2, 3, 1) #(B*T,128,128,2)
    return deformation


def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """
    b, c, h, w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed


def warping(source_image, deformation):
    r"""warp the input image according to the deformation
    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """
    _, h_old, w_old, _ = deformation.shape # B 128 128 2
    _, _, h, w = source_image.shape# B 32 128 128
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2) #B 2 128 128
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation)


class DenseFlowNetwork(torch.nn.Module):
    def __init__(self, num_channel=6, num_channel_modulation=3*5, hidden_size=256):
        super(DenseFlowNetwork, self).__init__()

        # Convolutional Layers
        self.conv1 = torch.nn.Conv2d(num_channel, 32, kernel_size=7, stride=1, padding=3)
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=32, affine=True)
        self.conv1_relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(32, 256, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=256, affine=True)
        self.conv2_relu = torch.nn.ReLU()


        # SPADE Blocks
        self.spade_layer_1 = SPADE(256, num_channel_modulation, hidden_size)
        self.spade_layer_2 = SPADE(256, num_channel_modulation, hidden_size)
        self.pixel_shuffle_1 = torch.nn.PixelShuffle(2)
        self.spade_layer_4 = SPADE(64, num_channel_modulation, hidden_size)

        # Final Convolutional Layer
        self.conv_4 = torch.nn.Conv2d(64, 2, kernel_size=7, stride=1, padding=3)
        self.conv_5= nn.Sequential(torch.nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(32, 1, kernel_size=7, stride=1, padding=3),
                                   torch.nn.Sigmoid(),
                                   )#predict weight

    def forward(self, ref_N_frame_img, ref_N_frame_sketch, T_driving_sketch): #to output: (B*T,3,H,W)
                   #   (B, N, 3, H, W)(B, N, 3, H, W)    (B, 5, 3, H, W)  #
        ref_N = ref_N_frame_img.size(1)

        driving_sketch=torch.cat([T_driving_sketch[:,i] for i in range(T_driving_sketch.size(1))], dim=1)  #(B, 3*5, H, W)

        wrapped_h1_sum, wrapped_h2_sum, wrapped_ref_sum=0.,0.,0.
        softmax_denominator=0.
        T = 1  # during rendering, generate T=1 image  at a time
        for ref_idx in range(ref_N): # each ref img provide information for each B*T frame
            ref_img= ref_N_frame_img[:, ref_idx]  #(B, 3, H, W)
            ref_img = ref_img.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B,T, 3, H, W)
            ref_img = torch.cat([ref_img[i] for i in range(ref_img.size(0))], dim=0)  # (B*T, 3, H, W)

            ref_sketch = ref_N_frame_sketch[:, ref_idx] #(B, 3, H, W)
            ref_sketch = ref_sketch.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B,T, 3, H, W)
            ref_sketch = torch.cat([ref_sketch[i] for i in range(ref_sketch.size(0))], dim=0)  # (B*T, 3, H, W)

            #predict flow and weight
            flow_module_input = torch.cat((ref_img, ref_sketch), dim=1)  #(B*T, 3+3, H, W)
            # Convolutional Layers
            h1 = self.conv1_relu(self.conv1_bn(self.conv1(flow_module_input)))   #(32,128,128)
            h2 = self.conv2_relu(self.conv2_bn(self.conv2(h1)))    #(256,64,64)
            # SPADE Blocks
            downsample_64 = downsample(driving_sketch, (64, 64))   # driving_sketch:(B*T, 3, H, W) B*T 3 64 64

            spade_layer = self.spade_layer_1(h2, downsample_64)  #(256,64,64)
            spade_layer = self.spade_layer_2(spade_layer, downsample_64)   #(256,64,64)

            spade_layer = self.pixel_shuffle_1(spade_layer)   #(64,128,128)

            spade_layer = self.spade_layer_4(spade_layer, driving_sketch)    #(64,128,128)

            # Final Convolutional Layer
            output_flow = self.conv_4(spade_layer)      #   (B*T,2,128,128)
            output_weight=self.conv_5(spade_layer)       #  (B*T,1,128,128)

            deformation=convert_flow_to_deformation(output_flow) # (B*T 128 128 2)
            wrapped_h1 = warping(h1, deformation)  #(32,128,128)
            wrapped_h2 = warping(h2, deformation)   #(256,64,64)
            wrapped_ref = warping(ref_img, deformation)  #(3,128,128)

            softmax_denominator+=output_weight
            wrapped_h1_sum+=wrapped_h1*output_weight
            wrapped_h2_sum+=wrapped_h2*downsample(output_weight, (64,64))
            wrapped_ref_sum+=wrapped_ref*output_weight
        #return weighted warped feataure and images
        softmax_denominator+=0.00001
        wrapped_h1_sum=wrapped_h1_sum/softmax_denominator
        wrapped_h2_sum = wrapped_h2_sum / downsample(softmax_denominator, (64,64))
        wrapped_ref_sum = wrapped_ref_sum / softmax_denominator
        return wrapped_h1_sum, wrapped_h2_sum, wrapped_ref_sum


class TranslationNetwork(torch.nn.Module):
    def __init__(self):
        super(TranslationNetwork, self).__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        # Encoder
        self.conv1 = torch.nn.Conv2d(in_channels=3+3*5, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=32, affine=True)
        self.conv1_relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=256, affine=True)
        self.conv2_relu = torch.nn.ReLU()

        # Decoder
        self.spade_1 = SPADE(num_channel=256, num_channel_modulation=256)
        self.adain_1 = AdaIN(256,512)
        self.pixel_suffle_1 = nn.PixelShuffle(upscale_factor=2)

        self.spade_2 = SPADE(num_channel=64, num_channel_modulation=32)
        self.adain_2 = AdaIN(input_channel=64,modulation_channel=512)

        self.spade_4 = SPADE(num_channel=64, num_channel_modulation=3)

        # Final layer
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)
        self.Sigmoid=torch.nn.Sigmoid()
    def forward(self, translation_input, wrapped_ref, wrapped_h1, wrapped_h2, T_mels):
        #              (B,3+3*5,H,W)   (B,3,128,128)  (B,32,128,128) (B,256,64,64) (B,T,1,h,w)  #T=1
        # Encoder
        T_mels=torch.cat([T_mels[i] for i in range(T_mels.size(0))],dim=0)# B*T,1,h,w
        x = self.conv1_relu(self.conv1_bn(self.conv1(translation_input)))    #32,128,128
        x = self.conv2_relu(self.conv2_bn(self.conv2(x)))  #256,64,64

        audio_feature = self.audio_encoder(T_mels).squeeze(-1).permute(0,2,1) #(B*T,1,512)

        # Decoder
        x = self.spade_1(x, wrapped_h2) # (C=256,64,64)
        x = self.adain_1(x, audio_feature)  # (C=256,64,64)
        x = self.pixel_suffle_1(x)   # (C=64,128,128)

        x = self.spade_2(x, wrapped_h1)   # (64,128,128)
        x = self.adain_2(x, audio_feature)  # (64,128,128)
        x = self.spade_4(x, wrapped_ref)    # (64,128,128)

        # output layer
        x = self.leaky_relu(x)
        x = self.conv_last(x)
        x = self.Sigmoid(x)
        return x

class Renderer(torch.nn.Module):
    def __init__(self):
        super(Renderer, self).__init__()

        # 1.flow Network
        self.flow_module = DenseFlowNetwork()
        #2. translation Network
        self.translation = TranslationNetwork()
        #3.return loss
        self.perceptual = PerceptualLoss(network='vgg19',
                                         layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                                         num_scales=2)

    def forward(self, face_frame_img, target_sketches, ref_N_frame_img, ref_N_frame_sketch, audio_mels): #T=1
        #            (B,1,3,H,W)   (B,5,3,H,W)       (B,N,3,H,W)   (B,N,3,H,W)  (B,T,1,hv,wv)T=1
        # (1)warping reference images and their feature
        wrapped_h1, wrapped_h2, wrapped_ref = self.flow_module(ref_N_frame_img, ref_N_frame_sketch, target_sketches)
        #(B,C,H,W)

        # (2)translation module
        target_sketches = torch.cat([target_sketches[:, i] for i in range(target_sketches.size(1))], dim=1)
        # (B,3*T,H,W)
        gt_face = torch.cat([face_frame_img[i] for i in range(face_frame_img.size(0))], dim=0)
        # (B,3,H,W)
        gt_mask_face = gt_face.clone()
        gt_mask_face[:, :, gt_mask_face.size(2) // 2:, :] = 0  # (B,3,H,W)
        #
        translation_input=torch.cat([gt_mask_face, target_sketches], dim=1) #  (B,3+3*5,H,W)
        generated_face = self.translation(translation_input, wrapped_ref, wrapped_h1, wrapped_h2, audio_mels) #translation_input

        perceptual_gen_loss = self.perceptual(generated_face, gt_face, use_style_loss=True,
                                              weight_style_to_perceptual=250).mean()
        perceptual_warp_loss = self.perceptual(wrapped_ref, gt_face, use_style_loss=False,
                                               weight_style_to_perceptual=0.).mean()
        return generated_face, wrapped_ref, torch.unsqueeze(perceptual_warp_loss, 0), torch.unsqueeze(
            perceptual_gen_loss, 0)
        # (B,3,H,W) and losses

#the following is the code for Perceptual(VGG) loss

def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output

class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output

def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)

class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.

    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the input images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None,
                 criterion='l1', resize=False, resize_mode='bilinear',
                 instance_normalized=False, num_scales=1,):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized


        print('Perceptual loss:')
        print('\tMode: {}'.format(network))

    def forward(self, inp, target, mask=None,use_style_loss=False,weight_style_to_perceptual=0.):
        r"""Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inp, target = \
            apply_imagenet_normalization(inp), \
            apply_imagenet_normalization(target)
        if self.resize:
            inp = F.interpolate(
                inp, mode=self.resize_mode, size=(256, 256),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(256, 256),
                align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0
        style_loss=0
        for scale in range(self.num_scales):
            input_features, target_features = \
                self.model(inp), self.model(target)
            for layer, weight in zip(self.layers, self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                if mask is not None:
                    mask_ = F.interpolate(mask, input_feature.shape[2:],
                                          mode='bilinear',
                                          align_corners=False)
                    input_feature = input_feature * mask_
                    target_feature = target_feature * mask_
                    # print('mask',mask_.shape)


                loss += weight * self.criterion(input_feature,
                                                target_feature)
                if use_style_loss and scale==0:
                    style_loss += self.criterion(self.compute_gram(input_feature),
                                                 self.compute_gram(target_feature))

            # Downsample the input and target.
            if scale != self.num_scales - 1:
                inp = F.interpolate(
                    inp, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        if use_style_loss:
            return loss + style_loss*weight_style_to_perceptual
        else:
            return loss


    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
```

#### 绘制landmark{#dl}

```py
"""MediaPipe solution drawing utils."""
import math
from typing import List, Mapping, Optional, Tuple, Union
import cv2
import dataclasses
import numpy as np
import tqdm
from mediapipe.framework.formats import landmark_pb2
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5 
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                 (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
#(10, 338), (338, 297), (297, 332), (332, 284),(284, 251), (251, 389) (162, 21), (21, 54),(54, 103), (103, 67), (67, 109), (109, 10)

FACEMESH_NOSE= frozenset([(168, 6),(6,197),(197,195),(195,5),(5,4),\
                          (4,45),(45,220),(220,115),(115,48),\
                          (4,275),(275,440),(440,344),(344,278),])
FACEMESH_FULL = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL,FACEMESH_NOSE
])
connections=FACEMESH_FULL

def summary_landmark(edge_set):
    landmarks=set()
    for a,b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks
all_landmark_idx=summary_landmark(FACEMESH_FULL)
pose_landmark_idx=\
summary_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW,FACEMESH_RIGHT_EYE,\
                                       FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,])).union([162,127,234,93,389,356,454,323])
content_landmark_idx= all_landmark_idx - pose_landmark_idx


def draw_landmarks(
    image: np.ndarray,
    landmark_list: List,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec()):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three channel BGR image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from
      hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius.
      If this argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from
      hand connections to the DrawingSpecs that specifies the
      connections' drawing settings such as color and line thickness.
      If this argument is explicitly set to None, no landmark connections will
      be drawn.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel BGR.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  if image.shape[2] != _BGR_CHANNELS:
    raise ValueError('Input image must contain three channel bgr data.')
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for landmark in landmark_list:
    # if ((landmark.HasField('visibility') and
    #      landmark.visibility < _VISIBILITY_THRESHOLD) or
    #     (landmark.HasField('presence') and
    #      landmark.presence < _PRESENCE_THRESHOLD)):
    #   continue
    idx=landmark.idx
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                 image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px

  if connections:
    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      # if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
      #   raise ValueError(f'Landmark index is out of range. Invalid connection '
      #                    f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        # if start_idx in content_landmark and end_idx in content_landmark:
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], drawing_spec.color,
                 drawing_spec.thickness)
  return image
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  # if landmark_drawing_spec:
  #   for idx, landmark_px in idx_to_coordinates.items():
  #     drawing_spec = landmark_drawing_spec[idx] if isinstance(
  #         landmark_drawing_spec, Mapping) else landmark_drawing_spec
  #     # White circle border
      # circle_border_radius = max(drawing_spec.circle_radius + 1,
      #                            int(drawing_spec.circle_radius * 1.2))
      # circle_border_radius=circle_border_radius*0.1
      # cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
      #            drawing_spec.thickness)
      # Fill color into the circle

      # cv2.circle(image, landmark_px, 1,
      #            drawing_spec.color, drawing_spec.thickness)
      # cv2.putText(image,str(idx),landmark_px,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
```

#### 推理{#is}

```py

import numpy as np
import cv2, os, argparse
import subprocess
from tqdm import tqdm
from models import Renderer
import torch
from models import Landmark_generator as Landmark_transformer
import face_alignment
from models import audio
from draw_landmark import draw_landmarks
import mediapipe as mp
parser = argparse.ArgumentParser()
parser.add_argument('--input', '--input_template_video', type=str, default='./test/template_video/129.mp4')
#'./test/template_video/129.mp4'

parser.add_argument('--audio', type=str, default='./test/template_video/audio2.wav')
#'./test/template_video/abstract.mp3'
#'./test/template_video/audio2.wav'
parser.add_argument('--output_dir', type=str, default='./test_result')
parser.add_argument('--static', type=bool, help='whether only use  the first frame for inference', default=False)
parser.add_argument('--landmark_gen_checkpoint_path', type=str, default='./test/checkpoints/landmarkgenerator_checkpoint.pth')
parser.add_argument('--renderer_checkpoint_path', type=str, default='./test/checkpoints/renderer_checkpoint.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parser.parse_args()

ref_img_N = 25
Nl = 15
T = 5
mel_step_size = 16
img_size = 128

mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
lip_index = [0, 17]  # the index of the midpoints of the upper lip and lower lip
landmark_gen_checkpoint_path = args.landmark_gen_checkpoint_path
renderer_checkpoint_path =args.renderer_checkpoint_path
output_dir = args.output_dir
temp_dir = 'tempfile_of_{}'.format(output_dir.split('/')[-1])
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
input_video_path = args.input
input_audio_path = args.audio

# the following is the index sequence for fical landmarks detected by mediapipe
ori_sequence_idx = [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
                    361, 323, 454, 356, 389,  #
                    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  #
                    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,  #
                    168, 6, 197, 195, 5,  #
                    48, 115, 220, 45, 4, 275, 440, 344, 278,  #
                    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  #
                    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,  #
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  #
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# the following is the connections of landmarks for drawing sketch image
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])
FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_CONNECTION = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])

full_face_landmark_sequence = [*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),  #upper-half face
                               *list(range(4, 21)),  # jaw
                               *list(range(91, 131))]  # mouth

def summarize_landmark(edge_set):  # summarize all ficial landmarks used to construct edge
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmarks_idx = summarize_landmark(FACEMESH_CONNECTION)
pose_landmark_idx = \
    summarize_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE,
                                             FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])
# pose landmarks are landmarks of the upper-half face(eyes,nose,cheek) that represents the pose information

content_landmark_idx = all_landmarks_idx - pose_landmark_idx
# content_landmark include landmarks of lip and jaw which are inferred from audio

if os.path.isfile(input_video_path) and input_video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

outfile_path = os.path.join(output_dir,
                       '{}_N_{}_Nl_{}.mp4'.format(input_video_path.split('/')[-1][:-4] + 'result', ref_img_N, Nl))
if os.path.isfile(input_video_path) and input_video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True


def swap_masked_region(target_img, src_img, mask): #function used in post-process
    """From src_img crop masked region to replace corresponding masked region
       in target_img
    """  # swap_masked_region(src_frame, generated_frame, mask=mask_img)
    mask_img = cv2.GaussianBlur(mask, (21, 21), 11)
    mask1 = mask_img / 255
    mask1 = np.tile(np.expand_dims(mask1, axis=2), (1, 1, 3))
    img = src_img * mask1 + target_img * (1 - mask1)
    return img.astype(np.uint8)

def merge_face_contour_only(src_frame, generated_frame, face_region_coord, fa): #function used in post-process
    """Merge the face from generated_frame into src_frame
    """
    input_img = src_frame
    y1, y2, x1, x2 = 0, 0, 0, 0
    if face_region_coord is not None:
        y1, y2, x1, x2 = face_region_coord
        input_img = src_frame[y1:y2, x1:x2]
    ### 1) Detect the facial landmarks
    preds = fa.get_landmarks(input_img)[0]  # 68x2
    if face_region_coord is not None:
        preds += np.array([x1, y1])
    lm_pts = preds.astype(int)
    contour_idx = list(range(0, 17)) + list(range(17, 27))[::-1]
    contour_pts = lm_pts[contour_idx]
    ### 2) Make the landmark region mark image
    mask_img = np.zeros((src_frame.shape[0], src_frame.shape[1], 1), np.uint8)
    cv2.fillConvexPoly(mask_img, contour_pts, 255)
    ### 3) Do swap
    img = swap_masked_region(src_frame, generated_frame, mask=mask_img)
    return img


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint
def load_model(model, path):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        if k[:6] == 'module':
            new_k=k.replace('module.', '', 1)
        else:
            new_k =k
        new_s[new_k] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

class LandmarkDict(dict):# Makes a dictionary that behave like an object to represent each landmark
    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
print(" landmark_generator_model loaded from : ", landmark_gen_checkpoint_path)
print(" renderer loaded from : ", renderer_checkpoint_path)
landmark_generator_model = load_model(
    model=Landmark_transformer(T=T, d_model=512, nlayers=4, nhead=4, dim_feedforward=1024, dropout=0.1),
    path=landmark_gen_checkpoint_path)
renderer = load_model(model=Renderer(), path=renderer_checkpoint_path)

##(1) Reading input video frames  ###
print('Reading video frames ... from', input_video_path)
if not os.path.isfile(input_video_path):
    raise ValueError('the input video file does not exist')
elif input_video_path.split('.')[1] in ['jpg', 'png', 'jpeg']: #if input a single image for testing
    ori_background_frames = [cv2.imread(input_video_path)]
else:
    video_stream = cv2.VideoCapture(input_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        print(" input video fps:", fps,',converting to 25fps...')
        command = 'ffmpeg -y -i ' + input_video_path + ' -r 25 ' + '{}/temp_25fps.avi'.format(temp_dir)
        subprocess.call(command, shell=True)
        input_video_path = '{}/temp_25fps.avi'.format(temp_dir)
        video_stream.release()
        video_stream = cv2.VideoCapture(input_video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
    assert fps == 25

    ori_background_frames = [] #input videos frames (includes background as well as face)
    frame_idx = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        ori_background_frames.append(frame)
        frame_idx = frame_idx + 1
input_vid_len = len(ori_background_frames)

##(2) Extracting audio####
if not input_audio_path.endswith('.wav'):
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(input_audio_path, '{}/temp.wav'.format(temp_dir))
    subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    input_audio_path = '{}/temp.wav'.format(temp_dir)
wav = audio.load_wav(input_audio_path, 16000)
mel = audio.melspectrogram(wav)  # (H,W)   extract mel-spectrum
##read audio mel into list###
mel_chunks = []  # each mel chunk correspond to 5 video frames, used to generate one video frame
mel_idx_multiplier = 80. / fps
mel_chunk_idx = 0
while 1:
    start_idx = int(mel_chunk_idx * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
        break
    mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])  # mel for generate one video frame
    mel_chunk_idx += 1
# mel_chunks = mel_chunks[:(len(mel_chunks) // T) * T]

##(3) detect facial landmarks using mediapipe tool
boxes = []  #bounding boxes of human face
lip_dists = [] #lip dists
#we define the lip dist(openness): distance between the  midpoints of the upper lip and lower lip
face_crop_results = []
all_pose_landmarks, all_content_landmarks = [], []  #content landmarks include lip and jaw landmarks
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5) as face_mesh:
    # (1) get bounding boxes and lip dist
    for frame_idx, full_frame in enumerate(ori_background_frames):
        h, w = full_frame.shape[0], full_frame.shape[1]
        results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise NotImplementedError  # not detect face
        face_landmarks = results.multi_face_landmarks[0]

        ## calculate the lip dist
        dx = face_landmarks.landmark[lip_index[0]].x - face_landmarks.landmark[lip_index[1]].x
        dy = face_landmarks.landmark[lip_index[0]].y - face_landmarks.landmark[lip_index[1]].y
        dist = np.linalg.norm((dx, dy))
        lip_dists.append((frame_idx, dist))

        # (1)get the marginal landmarks to crop face
        x_min,x_max,y_min,y_max = 999,-999,999,-999
        for idx, landmark in enumerate(face_landmarks.landmark):
            if idx in all_landmarks_idx:
                if landmark.x < x_min:
                    x_min = landmark.x
                if landmark.x > x_max:
                    x_max = landmark.x
                if landmark.y < y_min:
                    y_min = landmark.y
                if landmark.y > y_max:
                    y_max = landmark.y
        ##########plus some pixel to the marginal region##########
        #note:the landmarks coordinates returned by mediapipe range 0~1
        plus_pixel = 25
        x_min = max(x_min - plus_pixel / w, 0)
        x_max = min(x_max + plus_pixel / w, 1)

        y_min = max(y_min - plus_pixel / h, 0)
        y_max = min(y_max + plus_pixel / h, 1)
        y1, y2, x1, x2 = int(y_min * h), int(y_max * h), int(x_min * w), int(x_max * w)
        boxes.append([y1, y2, x1, x2])
    boxes = np.array(boxes)

    # (2)croppd face
    face_crop_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] \
                         for image, (y1, y2, x1, x2) in zip(ori_background_frames, boxes)]

    # (3)detect facial landmarks
    for frame_idx, full_frame in enumerate(ori_background_frames):
        h, w = full_frame.shape[0], full_frame.shape[1]
        results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("not detect face in some frame!")  # not detect
        face_landmarks = results.multi_face_landmarks[0]



        pose_landmarks, content_landmarks = [], []
        for idx, landmark in enumerate(face_landmarks.landmark):
            if idx in pose_landmark_idx:
                pose_landmarks.append((idx, w * landmark.x, h * landmark.y))
            if idx in content_landmark_idx:
                content_landmarks.append((idx, w * landmark.x, h * landmark.y))

        # normalize landmarks to 0~1
        y_min, y_max, x_min, x_max = face_crop_results[frame_idx][1]  #bounding boxes
        pose_landmarks = [ \
            [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in pose_landmarks]
        content_landmarks = [ \
            [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in content_landmarks]
        all_pose_landmarks.append(pose_landmarks)
        all_content_landmarks.append(content_landmarks)

# smooth landmarks
def get_smoothened_landmarks(all_landmarks, windows_T=1):
    for i in range(len(all_landmarks)):  # frame i
        if i + windows_T > len(all_landmarks):
            window = all_landmarks[len(all_landmarks) - windows_T:]
        else:
            window = all_landmarks[i: i + windows_T]
        #####
        for j in range(len(all_landmarks[i])):  # landmark j
            all_landmarks[i][j][1] = np.mean([frame_landmarks[j][1] for frame_landmarks in window])  # x
            all_landmarks[i][j][2] = np.mean([frame_landmarks[j][2] for frame_landmarks in window])  # y
    return all_landmarks

all_pose_landmarks = get_smoothened_landmarks(all_pose_landmarks, windows_T=1)
all_content_landmarks=get_smoothened_landmarks(all_content_landmarks,windows_T=1)


##randomly select N_l reference landmarks for landmark transformer##
dists_sorted = sorted(lip_dists, key=lambda x: x[1])
lip_dist_idx = np.asarray([idx for idx, dist in dists_sorted])  #the frame idxs sorted by lip openness

Nl_idxs = [lip_dist_idx[int(i)] for i in torch.linspace(0, input_vid_len - 1, steps=Nl)]
Nl_pose_landmarks, Nl_content_landmarks = [], []  #Nl_pose + Nl_content=Nl reference landmarks
for reference_idx in Nl_idxs:
    frame_pose_landmarks = all_pose_landmarks[reference_idx]
    frame_content_landmarks = all_content_landmarks[reference_idx]
    Nl_pose_landmarks.append(frame_pose_landmarks)
    Nl_content_landmarks.append(frame_content_landmarks)

Nl_pose = torch.zeros((Nl, 2, 74))  # 74 landmark
Nl_content = torch.zeros((Nl, 2, 57))  # 57 landmark
for idx in range(Nl):
    #arrange the landmark in a certain order, since the landmark index returned by mediapipe is is chaotic
    Nl_pose_landmarks[idx] = sorted(Nl_pose_landmarks[idx],
                                    key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
    Nl_content_landmarks[idx] = sorted(Nl_content_landmarks[idx],
                                       key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

    Nl_pose[idx, 0, :] = torch.FloatTensor(
        [Nl_pose_landmarks[idx][i][1] for i in range(len(Nl_pose_landmarks[idx]))])  # x
    Nl_pose[idx, 1, :] = torch.FloatTensor(
        [Nl_pose_landmarks[idx][i][2] for i in range(len(Nl_pose_landmarks[idx]))])  # y
    Nl_content[idx, 0, :] = torch.FloatTensor(
        [Nl_content_landmarks[idx][i][1] for i in range(len(Nl_content_landmarks[idx]))])  # x
    Nl_content[idx, 1, :] = torch.FloatTensor(
        [Nl_content_landmarks[idx][i][2] for i in range(len(Nl_content_landmarks[idx]))])  # y
Nl_content = Nl_content.unsqueeze(0)  # (1,Nl, 2, 57)
Nl_pose = Nl_pose.unsqueeze(0)  # (1,Nl,2,74)

##select reference images and draw sketches for rendering according to lip openness##
ref_img_idx = [int(lip_dist_idx[int(i)]) for i in torch.linspace(0, input_vid_len - 1, steps=ref_img_N)]
ref_imgs = [face_crop_results[idx][0] for idx in ref_img_idx]
## (N,H,W,3)
ref_img_pose_landmarks, ref_img_content_landmarks = [], []
for idx in ref_img_idx:
    ref_img_pose_landmarks.append(all_pose_landmarks[idx])
    ref_img_content_landmarks.append(all_content_landmarks[idx])

ref_img_pose = torch.zeros((ref_img_N, 2, 74))  # 74 landmark
ref_img_content = torch.zeros((ref_img_N, 2, 57))  # 57 landmark

for idx in range(ref_img_N):
    ref_img_pose_landmarks[idx] = sorted(ref_img_pose_landmarks[idx],
                                         key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
    ref_img_content_landmarks[idx] = sorted(ref_img_content_landmarks[idx],
                                            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
    ref_img_pose[idx, 0, :] = torch.FloatTensor(
        [ref_img_pose_landmarks[idx][i][1] for i in range(len(ref_img_pose_landmarks[idx]))])  # x
    ref_img_pose[idx, 1, :] = torch.FloatTensor(
        [ref_img_pose_landmarks[idx][i][2] for i in range(len(ref_img_pose_landmarks[idx]))])  # y

    ref_img_content[idx, 0, :] = torch.FloatTensor(
        [ref_img_content_landmarks[idx][i][1] for i in range(len(ref_img_content_landmarks[idx]))])  # x
    ref_img_content[idx, 1, :] = torch.FloatTensor(
        [ref_img_content_landmarks[idx][i][2] for i in range(len(ref_img_content_landmarks[idx]))])  # y

ref_img_full_face_landmarks = torch.cat([ref_img_pose, ref_img_content], dim=2).cpu().numpy()  # (N,2,131)
ref_img_sketches = []
for frame_idx in range(ref_img_full_face_landmarks.shape[0]):  # N
    full_landmarks = ref_img_full_face_landmarks[frame_idx]  # (2,131)
    h, w = ref_imgs[frame_idx].shape[0], ref_imgs[frame_idx].shape[1]
    drawn_sketech = np.zeros((int(h * img_size / min(h, w)), int(w * img_size / min(h, w)), 3))
    mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]], full_landmarks[0, idx],
                                               full_landmarks[1, idx]) for idx in range(full_landmarks.shape[1])]
    drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                   connection_drawing_spec=drawing_spec)
    drawn_sketech = cv2.resize(drawn_sketech, (img_size, img_size))  # (128, 128, 3)
    ref_img_sketches.append(drawn_sketech)
ref_img_sketches = torch.FloatTensor(np.asarray(ref_img_sketches) / 255.0).cuda().unsqueeze(0).permute(0, 1, 4, 2, 3)
# (1,N, 3, 128, 128)
ref_imgs = [cv2.resize(face.copy(), (img_size, img_size)) for face in ref_imgs]
ref_imgs = torch.FloatTensor(np.asarray(ref_imgs) / 255.0).unsqueeze(0).permute(0, 1, 4, 2, 3).cuda()
# (1,N,3,H,W)

##prepare output video strame##
frame_h, frame_w = ori_background_frames[0].shape[:-1]
out_stream = cv2.VideoWriter('{}/result.avi'.format(temp_dir), cv2.VideoWriter_fourcc(*'DIVX'), fps,
                             (frame_w * 2, frame_h))  # +frame_h*3


##generate final face image and output video##
input_mel_chunks_len = len(mel_chunks)
input_frame_sequence = torch.arange(input_vid_len).tolist()
#the input template video may be shorter than audio
#in this case we repeat the input template video as following
num_of_repeat=input_mel_chunks_len//input_vid_len+1
input_frame_sequence = input_frame_sequence + list(reversed(input_frame_sequence))
input_frame_sequence=input_frame_sequence*((num_of_repeat+1)//2)


for batch_idx, batch_start_idx in tqdm(enumerate(range(0, input_mel_chunks_len - 2, 1)),
                                       total=len(range(0, input_mel_chunks_len - 2, 1))):
    T_input_frame, T_ori_face_coordinates = [], []
    #note: input_frame include background as well as face
    T_mel_batch, T_crop_face,T_pose_landmarks = [], [],[]

    # (1) for each batch of T frame, generate corresponding landmarks using landmark generator
    for mel_chunk_idx in range(batch_start_idx, batch_start_idx + T):  # for each T frame
        # 1 input audio
        T_mel_batch.append(mel_chunks[max(0, mel_chunk_idx - 2)])

        # 2.input face
        input_frame_idx = int(input_frame_sequence[mel_chunk_idx])
        face, coords = face_crop_results[input_frame_idx]
        T_crop_face.append(face)
        T_ori_face_coordinates.append((face, coords))  ##input face
        # 3.pose landmarks
        T_pose_landmarks.append(all_pose_landmarks[input_frame_idx])
        # 3.background
        T_input_frame.append(ori_background_frames[input_frame_idx].copy())
    T_mels = torch.FloatTensor(np.asarray(T_mel_batch)).unsqueeze(1).unsqueeze(0)  # 1,T,1,h,w
    #prepare pose landmarks
    T_pose = torch.zeros((T, 2, 74))  # 74 landmark
    for idx in range(T):
        T_pose_landmarks[idx] = sorted(T_pose_landmarks[idx],
                                       key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        T_pose[idx, 0, :] = torch.FloatTensor(
            [T_pose_landmarks[idx][i][1] for i in range(len(T_pose_landmarks[idx]))])  # x
        T_pose[idx, 1, :] = torch.FloatTensor(
            [T_pose_landmarks[idx][i][2] for i in range(len(T_pose_landmarks[idx]))])  # y
    T_pose = T_pose.unsqueeze(0)  # (1,T, 2,74)

    #landmark  generator inference
    Nl_pose, Nl_content = Nl_pose.cuda(), Nl_content.cuda() # (Nl,2,74)  (Nl,2,57)
    T_mels, T_pose = T_mels.cuda(), T_pose.cuda()
    with torch.no_grad():  # require    (1,T,1,hv,wv)(1,T,2,74)(1,T,2,57)
        predict_content = landmark_generator_model(T_mels, T_pose, Nl_pose, Nl_content)  # (1*T,2,57)
    T_pose = torch.cat([T_pose[i] for i in range(T_pose.size(0))], dim=0)  # (1*T,2,74)
    T_predict_full_landmarks = torch.cat([T_pose, predict_content], dim=2).cpu().numpy()  # (1*T,2,131)

    #1.draw target sketch
    T_target_sketches = []
    for frame_idx in range(T):
        full_landmarks = T_predict_full_landmarks[frame_idx]  # (2,131)
        h, w = T_crop_face[frame_idx].shape[0], T_crop_face[frame_idx].shape[1]
        drawn_sketech = np.zeros((int(h * img_size / min(h, w)), int(w * img_size / min(h, w)), 3))
        mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]]
                                                   , full_landmarks[0, idx], full_landmarks[1, idx]) for idx in
                                      range(full_landmarks.shape[1])]
        drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                       connection_drawing_spec=drawing_spec)
        drawn_sketech = cv2.resize(drawn_sketech, (img_size, img_size))  # (128, 128, 3)
        if frame_idx == 2:
            show_sketch = cv2.resize(drawn_sketech, (frame_w, frame_h)).astype(np.uint8)
        T_target_sketches.append(torch.FloatTensor(drawn_sketech) / 255)
    T_target_sketches = torch.stack(T_target_sketches, dim=0).permute(0, 3, 1, 2)  # (T,3,128, 128)
    target_sketches = T_target_sketches.unsqueeze(0).cuda()  # (1,T,3,128, 128)

    # 2.lower-half masked face
    ori_face_img = torch.FloatTensor(cv2.resize(T_crop_face[2], (img_size, img_size)) / 255).permute(2, 0, 1).unsqueeze(
        0).unsqueeze(0).cuda()  #(1,1,3,H, W)

    # 3. render the full face
    # require (1,1,3,H,W)   (1,T,3,H,W)  (1,N,3,H,W)   (1,N,3,H,W)  (1,1,1,h,w)
    # return  (1,3,H,W)
    with torch.no_grad():
        generated_face, _, _, _ = renderer(ori_face_img, target_sketches, ref_imgs, ref_img_sketches,
                                                    T_mels[:, 2].unsqueeze(0))  # T=1
    gen_face = (generated_face.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H,W,3)

    # 4. paste each generated face
    y1, y2, x1, x2 = T_ori_face_coordinates[2][1]  # coordinates of face bounding box
    original_background = T_input_frame[2].copy()
    T_input_frame[2][y1:y2, x1:x2] = cv2.resize(gen_face,(x2 - x1, y2 - y1))  #resize and paste generated face
    # 5. post-process
    full = merge_face_contour_only(original_background, T_input_frame[2], T_ori_face_coordinates[2][1],fa)   #(H,W,3)
    # 6.output
    full = np.concatenate([show_sketch, full], axis=1)
    out_stream.write(full)
    if batch_idx == 0:
        out_stream.write(full)
        out_stream.write(full)
out_stream.release()
command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio_path, '{}/result.avi'.format(temp_dir), outfile_path)
subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print("succeed output results to:", outfile_path)

```
计算得分略，详见源代码

#### Wav2Lip{#W2l}

#### 面部检测模型{#fd}

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(nn.Module):

    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs


class ResNetDepth(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = nn.Conv2d(3 + 68, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
```

#### conv{#conv}

```py
import torch
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
```

#### syncnet{#sync}

```py
import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
```

#### wav2Lip{#w2l}

```py
import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs

class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),
            
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)

```

#### gen_videos_from_filelist{#gvff}

```py
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import dlib, json, subprocess
from tqdm import tqdm
from glob import glob
import torch

sys.path.append('../')
import audio
import face_detection
from models import Wav2Lip

parser = argparse.ArgumentParser(description='Code to generate results for test filelists')

parser.add_argument('--filelist', type=str, 
					help='Filepath of filelist file to read', required=True)
parser.add_argument('--results_dir', type=str, help='Folder to save all results into', 
									required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0], 
					help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, 
					help='Single GPU batch size for face detection', default=64)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip', default=128)

# parser.add_argument('--resize_factor', default=1, type=int)

args = parser.parse_args()
args.img_size = 96

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU')
			batch_size //= 2
			args.face_det_batch_size = batch_size
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			raise ValueError('Face not detected!')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = get_smoothened_boxes(np.array(results), T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results 

def datagen(frames, face_det_results, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		if i >= len(frames): raise ValueError('Equal or less lengths only')

		frame_to_save = frames[i].copy()
		face, coords, valid_frame = face_det_results[i].copy()
		if not valid_frame:
			continue

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

fps = 25
mel_step_size = 16
mel_idx_multiplier = 80./fps
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

model = load_model(args.checkpoint_path)

def main():
	assert args.data_root is not None
	data_root = args.data_root

	if not os.path.isdir(args.results_dir): os.makedirs(args.results_dir)

	with open(args.filelist, 'r') as filelist:
		lines = filelist.readlines()

	for idx, line in enumerate(tqdm(lines)):
		audio_src, video = line.strip().split()

		audio_src = os.path.join(data_root, audio_src) + '.mp4'
		video = os.path.join(data_root, video) + '.mp4'

		command = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'.format(audio_src, '../temp/temp.wav')
		subprocess.call(command, shell=True)
		temp_audio = '../temp/temp.wav'

		wav = audio.load_wav(temp_audio, 16000)
		mel = audio.melspectrogram(wav)
		if np.isnan(mel.reshape(-1)).sum() > 0:
			continue

		mel_chunks = []
		i = 0
		while 1:
			start_idx = int(i * mel_idx_multiplier)
			if start_idx + mel_step_size > len(mel[0]):
				break
			mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
			i += 1

		video_stream = cv2.VideoCapture(video)
			
		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading or len(full_frames) > len(mel_chunks):
				video_stream.release()
				break
			full_frames.append(frame)

		if len(full_frames) < len(mel_chunks):
			continue

		full_frames = full_frames[:len(mel_chunks)]

		try:
			face_det_results = face_detect(full_frames.copy())
		except ValueError as e:
			continue

		batch_size = args.wav2lip_batch_size
		gen = datagen(full_frames.copy(), face_det_results, mel_chunks)

		for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
			if i == 0:
				frame_h, frame_w = full_frames[0].shape[:-1]
				out = cv2.VideoWriter('../temp/result.avi', 
								cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)
					

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
			
			for pl, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				pl = cv2.resize(pl.astype(np.uint8), (x2 - x1, y2 - y1))
				f[y1:y2, x1:x2] = pl
				out.write(f)

		out.release()

		vid = os.path.join(args.results_dir, '{}.mp4'.format(idx))

		command = 'ffmpeg -loglevel panic -y -i {} -i {} -strict -2 -q:v 1 {}'.format(temp_audio, 
								'../temp/result.avi', vid)
		subprocess.call(command, shell=True)

if __name__ == '__main__':
	main()

```

#### real_videos_inference{#rvi}

```py
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import dlib, json, subprocess
from tqdm import tqdm
from glob import glob
import torch

sys.path.append('../')
import audio
import face_detection
from models import Wav2Lip

parser = argparse.ArgumentParser(description='Code to generate results on ReSyncED evaluation set')

parser.add_argument('--mode', type=str, 
					help='random | dubbed | tts', required=True)

parser.add_argument('--filelist', type=str, 
					help='Filepath of filelist file to read', default=None)

parser.add_argument('--results_dir', type=str, help='Folder to save all results into', 
									required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right)')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Single GPU batch size for face detection', default=16)

parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip', default=128)
parser.add_argument('--face_res', help='Approximate resolution of the face at which to test', default=180)
parser.add_argument('--min_frame_res', help='Do not downsample further below this frame resolution', default=480)
parser.add_argument('--max_frame_res', help='Downsample to at least this frame resolution', default=720)
# parser.add_argument('--resize_factor', default=1, type=int)

args = parser.parse_args()
args.img_size = 96

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def rescale_frames(images):
	rect = detector.get_detections_for_batch(np.array([images[0]]))[0]
	if rect is None:
		raise ValueError('Face not detected!')
	h, w = images[0].shape[:-1]

	x1, y1, x2, y2 = rect

	face_size = max(np.abs(y1 - y2), np.abs(x1 - x2))

	diff = np.abs(face_size - args.face_res)
	for factor in range(2, 16):
		downsampled_res = face_size // factor
		if min(h//factor, w//factor) < args.min_frame_res: break 
		if np.abs(downsampled_res - args.face_res) >= diff: break

	factor -= 1
	if factor == 1: return images

	return [cv2.resize(im, (im.shape[1]//(factor), im.shape[0]//(factor))) for im in images]


def face_detect(images):
	batch_size = args.face_det_batch_size
	images = rescale_frames(images)

	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			raise ValueError('Face not detected!')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = get_smoothened_boxes(np.array(results), T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results, images 

def datagen(frames, face_det_results, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		if i >= len(frames): raise ValueError('Equal or less lengths only')

		frame_to_save = frames[i].copy()
		face, coords, valid_frame = face_det_results[i].copy()
		if not valid_frame:
			continue

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

def increase_frames(frames, l):
	## evenly duplicating frames to increase length of video
	while len(frames) < l:
		dup_every = float(l) / len(frames)

		final_frames = []
		next_duplicate = 0.

		for i, f in enumerate(frames):
			final_frames.append(f)

			if int(np.ceil(next_duplicate)) == i:
				final_frames.append(f)

			next_duplicate += dup_every

		frames = final_frames

	return frames[:l]

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

model = load_model(args.checkpoint_path)

def main():
	if not os.path.isdir(args.results_dir): os.makedirs(args.results_dir)

	if args.mode == 'dubbed':
		files = listdir(args.data_root)
		lines = ['{} {}'.format(f, f) for f in files]

	else:
		assert args.filelist is not None
		with open(args.filelist, 'r') as filelist:
			lines = filelist.readlines()

	for idx, line in enumerate(tqdm(lines)):
		video, audio_src = line.strip().split()

		audio_src = os.path.join(args.data_root, audio_src)
		video = os.path.join(args.data_root, video)

		command = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'.format(audio_src, '../temp/temp.wav')
		subprocess.call(command, shell=True)
		temp_audio = '../temp/temp.wav'

		wav = audio.load_wav(temp_audio, 16000)
		mel = audio.melspectrogram(wav)

		if np.isnan(mel.reshape(-1)).sum() > 0:
			raise ValueError('Mel contains nan!')

		video_stream = cv2.VideoCapture(video)

		fps = video_stream.get(cv2.CAP_PROP_FPS)
		mel_idx_multiplier = 80./fps

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break

			if min(frame.shape[:-1]) > args.max_frame_res:
				h, w = frame.shape[:-1]
				scale_factor = min(h, w) / float(args.max_frame_res)
				h = int(h/scale_factor)
				w = int(w/scale_factor)

				frame = cv2.resize(frame, (w, h))
			full_frames.append(frame)

		mel_chunks = []
		i = 0
		while 1:
			start_idx = int(i * mel_idx_multiplier)
			if start_idx + mel_step_size > len(mel[0]):
				break
			mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
			i += 1

		if len(full_frames) < len(mel_chunks):
			if args.mode == 'tts':
				full_frames = increase_frames(full_frames, len(mel_chunks))
			else:
				raise ValueError('#Frames, audio length mismatch')

		else:
			full_frames = full_frames[:len(mel_chunks)]

		try:
			face_det_results, full_frames = face_detect(full_frames.copy())
		except ValueError as e:
			continue

		batch_size = args.wav2lip_batch_size
		gen = datagen(full_frames.copy(), face_det_results, mel_chunks)

		for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
			if i == 0:
				frame_h, frame_w = full_frames[0].shape[:-1]

				out = cv2.VideoWriter('../temp/result.avi', 
								cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)
					

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
			
			for pl, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				pl = cv2.resize(pl.astype(np.uint8), (x2 - x1, y2 - y1))
				f[y1:y2, x1:x2] = pl
				out.write(f)

		out.release()

		vid = os.path.join(args.results_dir, '{}.mp4'.format(idx))
		command = 'ffmpeg -loglevel panic -y -i {} -i {} -strict -2 -q:v 1 {}'.format('../temp/temp.wav', 
								'../temp/result.avi', vid)
		subprocess.call(command, shell=True)


if __name__ == '__main__':
	main()

```

计算得分略，详情参见源代码
