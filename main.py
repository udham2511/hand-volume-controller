import mediapipe
import math
import cv2
import numpy

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER


def overlay(
    background: numpy.ndarray, foreground: numpy.ndarray, point: tuple[int]
) -> None:
    """overlay PNG image to another png image

    Args:
        background (numpy.ndarray): background image
        foreground (numpy.ndarray): foreground image
        point (tuple[int]): position of foreground image
    """
    FGX = max(0, point[0] * -1)
    BGX = max(0, point[0])
    BGY = max(0, point[1])
    FGY = max(0, point[1] * -1)

    BGH, BGW = background.shape[:2]
    FGH, FGW = foreground.shape[:2]

    W = min(FGW, BGW, FGW + point[0], BGW - point[0])
    H = min(FGH, BGH, FGH + point[1], BGH - point[1])

    foreground = foreground[FGY : FGY + H, FGX : FGX + W]
    backgroundSubSection = background[BGY : BGY + H, BGX : BGX + W]

    alphaMask = numpy.dstack(tuple(foreground[:, :, 3] / 255.0 for _ in range(3)))

    background[BGY : BGY + H, BGX : BGX + W] = (
        backgroundSubSection * (1 - alphaMask) + foreground[:, :, :3] * alphaMask
    )


def isFingerUP(
    p1: int, p2: int, p3: int, p4: int, landmarks, threshold: int = 0.65
) -> bool:
    """checks if finger is up or not

    Args:
        p1 (int): index of p1 position
        p2 (int): index of p2 position
        p3 (int): index of p3 position
        p4 (int): index of p4 position
        landmarks: landmarks of hand
        threshold (float): threshold level

    Returns:
        bool: finger up or not
    """
    P12P2 = toVector(landmarks[p1], landmarks[p2])
    P32P4 = toVector(landmarks[p3], landmarks[p4])

    P12P2DOTP32P4 = numpy.dot(P12P2, P32P4)

    normalisedP12P2DOTP32P4 = numpy.linalg.norm(P12P2) * numpy.linalg.norm(P32P4)

    return (
        P12P2DOTP32P4 / normalisedP12P2DOTP32P4 if normalisedP12P2DOTP32P4 != 0 else 0
    ) > threshold


toVector = lambda p1, p2: numpy.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

NOVOLUMEIMAGE = cv2.imread(r"./image/no-volume.png", cv2.IMREAD_UNCHANGED)
SMOOTHNESS = 2

drawing = mediapipe.solutions.drawing_utils
styles = mediapipe.solutions.drawing_styles

hands = mediapipe.solutions.hands

landmarks = numpy.zeros((21, 2), numpy.int32)

DEVICES = AudioUtilities.GetSpeakers()
INTERFACE = DEVICES.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
speakers = cast(INTERFACE, POINTER(IAudioEndpointVolume))

oldVolume = int(speakers.GetChannelVolumeLevelScalar(0) * 100)

SHAPE = (1280, 720)

cap = cv2.VideoCapture(0)

cap.set(3, SHAPE[0])
cap.set(4, SHAPE[1])

with hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as detector:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        cv2.flip(frame, 1, frame)

        frame.flags.writeable = False

        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS,
                    styles.DrawingSpec((0, 0, 255), -1, 6),
                    styles.DrawingSpec((0, 255, 0), 3, -1),
                )

                for num, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[num][0] = landmark.x * frame.shape[1]
                    landmarks[num][1] = landmark.y * frame.shape[0]

                for num in (4, 8):
                    cv2.circle(frame, landmarks[num], 8, (255, 255, 0), -1)
                    cv2.circle(frame, landmarks[num], 20, (255, 255, 0), 6)

                cv2.line(frame, landmarks[4], landmarks[8], (255, 255, 0), 7)

                distance = math.hypot(*(landmarks[4] - landmarks[8]))
                reference = math.hypot(*(landmarks[0] - landmarks[5]))

                volume = numpy.interp(distance, (50, reference), (0, 100))
                volume = SMOOTHNESS * round(volume / SMOOTHNESS)

                volume = 0.70 * oldVolume + 0.30 * volume

                volume = math.floor(volume) if volume < 50 else round(volume) + 1

                if volume > 100:
                    volume = 100

                if not isFingerUP(6, 8, 10, 12, hand_landmarks.landmark):
                    speakers.SetMasterVolumeLevelScalar(volume / 100.0, None)

                oldVolume = volume

        if oldVolume == 0:
            overlay(frame, NOVOLUMEIMAGE, (1170, 10))

        x = int(numpy.interp(oldVolume, (0, 100), (630, 80)))

        cv2.rectangle(frame, (60, x), (105, 610), (0, 138, 255), -1)
        cv2.ellipse(frame, (85, x), (22, 22), 180, 0, 180, (0, 138, 255), -1)

        cv2.ellipse(frame, (85, 80), (22, 22), 180, 0, 180, (255, 255, 255), 7)

        cv2.line(frame, (63, 80), (63, 650), (255, 255, 255), 7)
        cv2.line(frame, (108, 80), (108, 650), (255, 255, 255), 7)

        cv2.circle(frame, (85, 640), 37, (0, 138, 255), -1)
        cv2.circle(frame, (85, 640), 40, (255, 255, 255), 6)

        cv2.putText(
            frame,
            str(oldVolume).zfill(3),
            (57, 650),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Hand Volume Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()

cv2.destroyAllWindows()
