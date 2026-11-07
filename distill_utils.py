"""Frame-count math shared by the teacher scripts, the training script and tests.

Pure Python (no torch / speechbrain), so the teacher env can import it too.

Student (SpeechBrain Fbank, hop 160, center=True -> N // 160 + 1 mel frames),
then the ConvolutionFrontEnd: two 3x3 convs with stride 2. With
`padding: causal`, SpeechBrain 1.0.0 pads the FREQUENCY axis only, so the
time axis is effectively 'valid':  T_out = (T_in - 3) // 2 + 1.
Receptive field: student frame i depends on mel frames 4i .. 4i + 6.

Teacher (Whisper): N // 160 mel frames (Whisper drops the last STFT frame),
conv2 stride 2 -> ceil(n / 2) encoder frames at 50 fps, then non-overlapping
pair-averaging to 25 fps -> n_enc // 2.
"""

import math

HOP = 160
CNN_KERNEL = 3
CNN_STRIDE = 2
CNN_LAYERS = 2
# right edge of the CNN receptive field, in mel frames, for student frame i:
#   last_mel(i) = 4 * i + RF_RIGHT
RF_RIGHT = 6
MEL_PER_STUDENT_FRAME = CNN_STRIDE ** CNN_LAYERS  # 4


def student_mel_frames(num_samples: int) -> int:
    return num_samples // HOP + 1


def cnn_out_frames(t_in: int, padding: str = "causal") -> int:
    """Time frames after the 2-block CNN front end."""
    t = t_in
    for _ in range(CNN_LAYERS):
        if padding == "causal":  # = valid in time for SpeechBrain 1.0.0
            t = (t - CNN_KERNEL) // CNN_STRIDE + 1
        elif padding == "same":
            t = math.ceil(t / CNN_STRIDE)
        else:
            raise ValueError(padding)
    return max(t, 0)


def student_frames(num_samples: int, padding: str = "causal") -> int:
    return cnn_out_frames(student_mel_frames(num_samples), padding)


def teacher_enc_frames_50hz(num_samples: int) -> int:
    return math.ceil((num_samples // HOP) / 2)


def teacher_frames(num_samples: int) -> int:
    return teacher_enc_frames_50hz(num_samples) // 2


def first_affected_student_frame(k: int) -> int:
    """Smallest student frame index whose receptive field reaches mel frame k."""
    return max(0, math.ceil((k - RF_RIGHT) / MEL_PER_STUDENT_FRAME))
