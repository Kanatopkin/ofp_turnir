import os
import shutil
import subprocess
import tempfile
import urllib.request
from collections import deque
import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


class LM:
    NOSE = 0
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


POSE_CONNECTIONS = [
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
]


def ensure_model_path():
    """Скачивает модель MediaPipe Pose Landmarker Full при первом запуске."""
    model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def draw_skeleton(frame, landmarks, chin_pt=None):
    """Отрисовывает ключевые точки, кости скелета и точку подбородка на кадре."""
    h, w, _ = frame.shape
    for idx_a, idx_b in POSE_CONNECTIONS:
        pt1 = (int(landmarks[idx_a].x * w), int(landmarks[idx_a].y * h))
        pt2 = (int(landmarks[idx_b].x * w), int(landmarks[idx_b].y * h))
        cv2.line(frame, pt1, pt2, (255, 200, 0), 3)

    for idx in [
        LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
        LM.LEFT_ELBOW, LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST,
        LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE,
        LM.LEFT_ANKLE, LM.RIGHT_ANKLE
    ]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

    if chin_pt is not None:
        ch_x, ch_y = int(chin_pt[0] * w), int(chin_pt[1] * h)
        cv2.circle(frame, (ch_x, ch_y), 7, (255, 0, 255), -1)
        cv2.putText(
            frame, "CHIN", (ch_x + 10, ch_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
        )


def draw_hud(frame, valid_reps, total_attempts, state, hang_sec, elbow_deg, knee_deg, bar_y, chin_above_now, active_errors):
    """Отрисовывает линию перекладины сверху и панель телеметрии (HUD) в левом нижнем углу."""
    h, w, _ = frame.shape

    if bar_y is not None:
        bar_px = int(bar_y * h)
        bar_color = (0, 255, 0) if chin_above_now else (0, 215, 255)
        cv2.line(frame, (0, bar_px), (w, bar_px), bar_color, 3)
        label_y = bar_px + 22 if bar_px < 35 else bar_px - 8
        cv2.putText(
            frame,
            f"BAR LINE (Y={bar_y:.2f}) {'[CHIN ABOVE]' if chin_above_now else ''}",
            (15, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2
        )

    overlay = frame.copy()
    box_w = min(w - 20, 440)
    box_h = 130
    y_top = max(10, h - box_h - 15)
    cv2.rectangle(overlay, (10, y_top), (10 + box_w, y_top + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    cv2.putText(
        frame, f"VALID REPS: {valid_reps} / ATTEMPTS: {total_attempts}",
        (20, y_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 120), 2
    )
    cv2.putText(
        frame, f"STATE: {state} | HANG: {hang_sec:.2f}s",
        (20, y_top + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"ELBOW: {int(elbow_deg)} deg | KNEE: {int(knee_deg)} deg",
        (20, y_top + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 230, 255), 2
    )
    err_status = "ERRORS IN REP: YES" if active_errors else "TECHNIQUE: OK"
    err_color = (0, 80, 255) if active_errors else (0, 255, 0)
    cv2.putText(
        frame, err_status,
        (20, y_top + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.56, err_color, 2
    )


def calculate_angle(a, b, c):
    """Вычисляет угол в градусах (0-180) в вершине b между точками a-b-c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def format_timestamp(seconds):
    """Форматирует время в вид ММ:СС.д"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


def compute_bar_y_from_hands(lm):
    """
    Вычисляет координату Y верхнего среза перекладины по костяшкам пальцев (INDEX/PINKY)
    и запястьям (WRIST), чтобы линия проходила точно по трубе турника, а не ниже кистей.
    """
    knuckle_y = min(
        (lm[LM.LEFT_INDEX].y + lm[LM.RIGHT_INDEX].y) / 2.0,
        (lm[LM.LEFT_PINKY].y + lm[LM.RIGHT_PINKY].y) / 2.0
    )
    wrist_y = (lm[LM.LEFT_WRIST].y + lm[LM.RIGHT_WRIST].y) / 2.0
    return float(min(knuckle_y, wrist_y - 0.018))


def get_calibration_preview(video_path, manual_bar_y=None):
    """
    Возвращает первый кадр видео с размеченной линией перекладины
    и автоматически определенную координату Y грифа (от 0.0 до 1.0).
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, 0.25

    h, w, _ = frame.shape
    detected_bar_y = 0.25
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    model_path = ensure_model_path()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_image)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            detected_bar_y = compute_bar_y_from_hands(lm)
            nose = [lm[LM.NOSE].x, lm[LM.NOSE].y]
            mouth_x = (lm[LM.MOUTH_LEFT].x + lm[LM.MOUTH_RIGHT].x) / 2.0
            mouth_y = (lm[LM.MOUTH_LEFT].y + lm[LM.MOUTH_RIGHT].y) / 2.0
            chin_y = mouth_y + 0.45 * abs(mouth_y - nose[1])
            draw_skeleton(frame, lm, chin_pt=(mouth_x, chin_y))

    bar_y = manual_bar_y if manual_bar_y is not None else detected_bar_y
    bar_px = int(bar_y * h)

    cv2.line(frame, (0, bar_px), (w, bar_px), (0, 255, 0), 3)
    cv2.putText(
        frame, f"Bar Line Y={bar_y:.2f}", (20, max(30, bar_px - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round(detected_bar_y, 3)


def convert_to_web_mp4(raw_video_path):
    """Конвертирует видео через ffmpeg в H.264 (yuv420p) для воспроизведения в браузере."""
    web_path = raw_video_path.replace(".mp4", "_web.mp4")
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", raw_video_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-crf", "26",
            "-movflags", "+faststart", web_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(web_path):
            os.remove(raw_video_path)
            return web_path
    return raw_video_path


def analyze_video_pose(
    video_path,
    manual_bar_y=None,
    elbow_ext_angle=162.0,
    min_hang_sec=0.90,
    generate_tracked_video=True,
    progress_callback=None
):
    """
    Покадровый анализ видео на основе MediaPipe PoseLandmarker и конечного автомата (State Machine)
    с точной детекцией:
    - недотяга подбородка относительно верхнего среза грифа,
    - сгибания коленей (включая захлест голеней назад при фронтальной съемке и в висе перед тягой),
    - скрещивания и разведения ног,
    - рывков/махов ногами и туловищем,
    - поочередного (неравномерного) сгибания рук.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(1.0, 720.0 / max(orig_w, orig_h, 1))
    out_w = max(2, int(orig_w * scale) // 2 * 2)
    out_h = max(2, int(orig_h * scale) // 2 * 2)

    writer = None
    raw_out_path = None
    if generate_tracked_video:
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        raw_out_path = tmp_out.name
        tmp_out.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_out_path, fourcc, fps, (out_w, out_h))

    state = "WAIT_HANG"  # WAIT_HANG -> PULLING -> DESCENDING
    hang_start_time = None
    last_hang_duration = 0.0

    bar_y_samples = []
    shoulder_w_samples = []
    bar_y = manual_bar_y
    base_shoulder_w = 0.12

    # Сглаживающие буферы
    chin_buf = deque(maxlen=3)
    elbow_buf = deque(maxlen=3)
    knee_buf = deque(maxlen=3)
    shoulder_y_buf = deque(maxlen=3)

    # Кольцевой буфер нарушений ИП в висе за последние 25 кадров (~0.8 с перед стартом тяги)
    pre_pull_knee_err = deque(maxlen=25)
    pre_pull_cross_err = deque(maxlen=25)

    rep_start_time = 0.0
    min_elbow_in_rep = 180.0
    min_shoulder_y_in_rep = 1.0
    min_chin_y_in_rep = 1.0
    chin_above_frames = 0
    rep_errors = set()

    rep_frames = 0
    asym_frames = 0
    knee_bent_frames = 0
    legs_crossed_frames = 0
    feet_apart_frames = 0
    kipping_frames = 0

    reps_log = []
    frame_idx = 0
    last_ts_ms = -1

    model_path = ensure_model_path()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = frame_idx / fps
            ts_ms = max(last_ts_ms + 1, int(round(current_time * 1000)))
            last_ts_ms = ts_ms

            if progress_callback and total_frames > 0 and frame_idx % 10 == 0:
                progress_callback(min(1.0, frame_idx / total_frames))

            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_image, ts_ms)

            mean_elbow_angle = 0.0
            mean_knee_angle = 180.0
            chin_above_now = False

            if results.pose_landmarks:
                lm = results.pose_landmarks[0]

                def pt(idx):
                    p = lm[idx]
                    return [p.x, p.y]

                l_sh, r_sh = pt(LM.LEFT_SHOULDER), pt(LM.RIGHT_SHOULDER)
                l_el, r_el = pt(LM.LEFT_ELBOW), pt(LM.RIGHT_ELBOW)
                l_wr, r_wr = pt(LM.LEFT_WRIST), pt(LM.RIGHT_WRIST)
                l_hip, r_hip = pt(LM.LEFT_HIP), pt(LM.RIGHT_HIP)
                l_kn, r_kn = pt(LM.LEFT_KNEE), pt(LM.RIGHT_KNEE)
                l_an, r_an = pt(LM.LEFT_ANKLE), pt(LM.RIGHT_ANKLE)

                nose = pt(LM.NOSE)
                m_l = pt(LM.MOUTH_LEFT)
                m_r = pt(LM.MOUTH_RIGHT)
                mouth_x = (m_l[0] + m_r[0]) / 2.0
                mouth_y = (m_l[1] + m_r[1]) / 2.0
                # Анатомическое положение нижней точки подбородка
                raw_chin_y = mouth_y + 0.45 * abs(mouth_y - nose[1])
                raw_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0

                wrist_y = (l_wr[1] + r_wr[1]) / 2.0
                wrist_x = (l_wr[0] + r_wr[0]) / 2.0
                if manual_bar_y is None and len(bar_y_samples) < 60:
                    bar_y_samples.append(compute_bar_y_from_hands(lm))
                    bar_y = float(np.median(bar_y_samples))

                angle_el_l = calculate_angle(l_sh, l_el, l_wr)
                angle_el_r = calculate_angle(r_sh, r_el, r_wr)
                raw_elbow = (angle_el_l + angle_el_r) / 2.0

                # 1. Расчет угла в коленях с учетом 3D-захлеста голеней назад (попытка № 5, 00:26-00:28)
                angle_kn_2d = (
                    calculate_angle(l_hip, l_kn, l_an) + calculate_angle(r_hip, r_kn, r_an)
                ) / 2.0

                thigh_len = (
                    np.hypot(l_kn[0] - l_hip[0], l_kn[1] - l_hip[1]) +
                    np.hypot(r_kn[0] - r_hip[0], r_kn[1] - r_hip[1])
                ) / 2.0
                shin_len = (
                    np.hypot(l_an[0] - l_kn[0], l_an[1] - l_kn[1]) +
                    np.hypot(r_an[0] - r_kn[0], r_an[1] - r_kn[1])
                ) / 2.0
                shin_ratio = shin_len / max(0.04, thigh_len)

                # Если голень в 2D-проекции укоротилась (< 0.68 длины бедра) — ноги согнуты в коленях назад
                if shin_ratio < 0.68:
                    sagittal_knee_deg = float(np.degrees(np.arcsin(np.clip(shin_ratio / 0.85, 0.1, 1.0))) * 1.6)
                    raw_knee = min(angle_kn_2d, sagittal_knee_deg)
                else:
                    raw_knee = angle_kn_2d

                chin_buf.append(raw_chin_y)
