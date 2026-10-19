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

# =====================================================================
# КОНФИГУРАТОР ПАРАМЕТРОВ И ОГРАНИЧЕНИЙ АЛГОРИТМА (П. 1 - 26)
# =====================================================================
CONFIG = {
    # 1. Фиксация ИП перед стартом (с)
    "min_hang_sec": 0.55,
    # 2. Угол локтей в нижнем висе (градусы)
    "elbow_ext_angle": 160.0,
    # 3. Вис на полусогнутых руках перед стартом (с)
    "pre_bend_max_time": 0.55,
    # 4. Порог начала движения / старт тяги (градусы)
    "pull_start_angle": 148.0,
    # 5. Минимальная глубина движения (градусы)
    "min_pull_depth_angle": 135.0,
    # 6. Минимальная длительность попытки (с)
    "min_rep_duration": 0.45,
    # 7. Допуск высоты подбородка относительно грифа (доля высоты кадра)
    "chin_bar_tolerance": 0.005,
    # 8. Максимальный угол локтей в пике (градусы)
    "max_peak_elbow_angle": 45.0,
    # 9. Высота носа над грифом (доля высоты кадра)
    "nose_bar_margin": -0.015,
    # 10. Число подтвержденных кадров пика
    "min_chin_above_frames": 1,
    # 11. Порог прямоты ног / 2D-угол в коленях (градусы)
    "knee_bent_threshold": 130.0,
    # 12. Порог захлеста голеней назад 3D (доля длины бедра)
    "shin_ratio_threshold": 0.66,
    # 13. Длительность ошибки согнутых коленей (25% кадров, мин. 5 кадров)
    "knee_err_frame_ratio": 0.25,
    "knee_err_min_frames": 5,
    # 14. Порог скрещивания ног / X-перехлест
    "cross_product_threshold": -0.00001,
    # 15. Длительность ошибки скрещивания (мин. 5 кадров)
    "cross_err_min_frames": 5,
    # 16. Разведение ног / ступни не вместе (125% ширины плеч)
    "feet_apart_ratio": 1.25,
    # 17. Длительность разведения ног (30% кадров, мин. 5 кадров)
    "feet_apart_frame_ratio": 0.30,
    "feet_apart_min_frames": 5,
    # 18. Рывок тазом / угол в тазобедренном суставе (градусы)
    "hip_kipping_angle": 156.0,
    # 19. Горизонтальный вынос таза (72% ширины плеч)
    "hip_swing_ratio": 0.72,
    # 20. Вынос стоп от таза / мах ногами (70% ширины плеч)
    "ankle_swing_ratio": 0.60,
    # 21. Длительность ошибки маха/рывка (30% кадров, мин. 6 кадров)
    "kipping_frame_ratio": 0.40,
    "kipping_min_frames": 6,
    # 22. Асимметрия: чистая разница углов в локтях сверх калибровки (градусы)
    "asym_elbow_diff_deg": 30.0,
    # 23. Асимметрия: перекос линии локтей (20% ширины плеч)
    "asym_elbow_tilt_ratio": 0.30,
    # 24. Асимметрия: перекос линии плеч (20% ширины плеч)
    "asym_shoulder_tilt_ratio": 0.30,
    # 25. Длительность поочередного сгибания рук (20% кадров, мин. 5 кадров)
    "asym_frame_ratio": 0.30,
    "asym_min_frames": 5,
    # 26. Отрыв кистей от грифа / срыв или спрыгивание (16% высоты кадра)
    "bar_release_dist": 0.16,
}


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
    model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle


def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


def compute_bar_y_from_hands(lm):
    knuckle_y = min(
        (lm[LM.LEFT_INDEX].y + lm[LM.RIGHT_INDEX].y) / 2.0,
        (lm[LM.LEFT_PINKY].y + lm[LM.RIGHT_PINKY].y) / 2.0
    )
    wrist_y = (lm[LM.LEFT_WRIST].y + lm[LM.RIGHT_WRIST].y) / 2.0
    return float(0.75 * knuckle_y + 0.25 * wrist_y)


def draw_skeleton(frame, landmarks, chin_pt=None):
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
        cv2.putText(frame, "CHIN", (ch_x + 10, ch_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


def draw_hud(frame, valid_reps, total_attempts, state, hang_sec, elbow_deg, knee_deg, bar_y, chin_above_now, active_errors):
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
    box_w, box_h = min(w - 20, 440), 130
    y_top = max(10, h - box_h - 15)
    cv2.rectangle(overlay, (10, y_top), (10 + box_w, y_top + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    cv2.putText(frame, f"VALID REPS: {valid_reps} / ATTEMPTS: {total_attempts}", (20, y_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 120), 2)
    cv2.putText(frame, f"STATE: {state} | HANG: {hang_sec:.2f}s", (20, y_top + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    cv2.putText(frame, f"ELBOW: {int(elbow_deg)} deg | KNEE: {int(knee_deg)} deg", (20, y_top + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 230, 255), 2)
    err_status = "ERRORS IN REP: YES" if active_errors else "TECHNIQUE: OK"
    err_color = (0, 80, 255) if active_errors else (0, 255, 0)
    cv2.putText(frame, err_status, (20, y_top + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.56, err_color, 2)


def get_calibration_preview(video_path, manual_bar_y=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, 0.25

    h, w, _ = frame.shape
    detected_bar_y = 0.25
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    model_path = ensure_model_path()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_image)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            detected_bar_y = compute_bar_y_from_hands(lm)
            nose = [lm[LM.NOSE].x, lm[LM.NOSE].y]
            mouth_x = (lm[LM.MOUTH_LEFT].x + lm[LM.MOUTH_RIGHT].x) / 2.0
            mouth_y = (lm[LM.MOUTH_LEFT].y + lm[LM.MOUTH_RIGHT].y) / 2.0
            chin_y = mouth_y + 0.30 * abs(mouth_y - nose[1])
            draw_skeleton(frame, lm, chin_pt=(mouth_x, chin_y))

    bar_y = manual_bar_y if manual_bar_y is not None else detected_bar_y
    bar_px = int(bar_y * h)
    cv2.line(frame, (0, bar_px), (w, bar_px), (0, 255, 0), 3)
    cv2.putText(frame, f"Bar Line Y={bar_y:.2f}", (20, max(30, bar_px - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round(detected_bar_y, 3)


def convert_to_web_mp4(raw_video_path):
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
    elbow_ext_angle=CONFIG["elbow_ext_angle"],
    min_hang_sec=CONFIG["min_hang_sec"],
    generate_tracked_video=True,
    progress_callback=None
):
    result = {
        "total_attempts": 0,
        "valid_reps": 0,
        "bar_y": round(manual_bar_y if manual_bar_y is not None else 0.12, 3),
        "errors": [],
        "reps_detail": [],
        "comment": "Повторения не зафиксированы.",
        "tracked_video_path": None
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["comment"] = "Ошибка открытия видеофайла."
        return result

    writer = None
    raw_out_path = None
    reps_log = []
    bar_y = manual_bar_y if manual_bar_y is not None else 0.12

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = min(1.0, 720.0 / max(orig_w, orig_h, 1))
        out_w = max(2, int(orig_w * scale) // 2 * 2)
        out_h = max(2, int(orig_h * scale) // 2 * 2)

        if generate_tracked_video:
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            raw_out_path = tmp_out.name
            tmp_out.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_out_path, fourcc, fps, (out_w, out_h))

        state = "WAIT_HANG"
        hang_start_time = None
        last_hang_duration = 0.0
        pre_bend_frames = 0

        bar_y_samples = []
        shoulder_w_samples = []
        base_sh_tilt_samples = []
        base_el_tilt_samples = []
        base_el_diff_samples = []
        base_head_off_samples = []
        base_body_len_samples = []

        base_shoulder_w = 0.12
        base_sh_tilt = 0.03
        base_el_tilt = 0.03
        base_el_diff = 5.0
        base_head_off = 0.0
        base_body_len = 0.50

        # Сглаживающие буферы по 3 кадрам
        chin_buf = deque(maxlen=3)
        nose_buf = deque(maxlen=3)
        elbow_buf = deque(maxlen=3)
        knee_buf = deque(maxlen=3)
        shoulder_y_buf = deque(maxlen=3)

        # Буферы контроля ИП в висе перед началом тяги
        pre_pull_knee_err = deque(maxlen=25)
        pre_pull_cross_err = deque(maxlen=25)
        pre_pull_swing_x = deque(maxlen=25)

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
        rep_ankle_x = []
        rep_hip_x = []

        frame_idx = 0
        last_ts_ms = -1

        model_path = ensure_model_path()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55
        )

        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
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
                        return [lm[idx].x, lm[idx].y]

                    l_sh, r_sh = pt(LM.LEFT_SHOULDER), pt(LM.RIGHT_SHOULDER)
                    l_el, r_el = pt(LM.LEFT_ELBOW), pt(LM.RIGHT_ELBOW)
                    l_wr, r_wr = pt(LM.LEFT_WRIST), pt(LM.RIGHT_WRIST)
                    l_hip, r_hip = pt(LM.LEFT_HIP), pt(LM.RIGHT_HIP)
                    l_kn, r_kn = pt(LM.LEFT_KNEE), pt(LM.RIGHT_KNEE)
                    l_an, r_an = pt(LM.LEFT_ANKLE), pt(LM.RIGHT_ANKLE)

                    nose = pt(LM.NOSE)
                    m_l, m_r = pt(LM.MOUTH_LEFT), pt(LM.MOUTH_RIGHT)
                    mouth_x = (m_l[0] + m_r[0]) / 2.0
                    mouth_y = (m_l[1] + m_r[1]) / 2.0
                    raw_chin_y = mouth_y + 0.30 * abs(mouth_y - nose[1])
                    raw_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0

                    wrist_y = (l_wr[1] + r_wr[1]) / 2.0
                    wrist_x = (l_wr[0] + r_wr[0]) / 2.0
                    if manual_bar_y is None and len(bar_y_samples) < 60:
                        bar_y_samples.append(compute_bar_y_from_hands(lm))
                        bar_y = float(np.median(bar_y_samples))

                    angle_el_l = calculate_angle(l_sh, l_el, l_wr)
                    angle_el_r = calculate_angle(r_sh, r_el, r_wr)
                    raw_elbow = (angle_el_l + angle_el_r) / 2.0

                    # П. 11-12: угол в коленях и 3D-захлест голеней назад
                    angle_kn_2d = (calculate_angle(l_hip, l_kn, l_an) + calculate_angle(r_hip, r_kn, r_an)) / 2.0
                    thigh_len = (np.hypot(l_kn[0] - l_hip[0], l_kn[1] - l_hip[1]) + np.hypot(r_kn[0] - r_hip[0], r_kn[1] - r_hip[1])) / 2.0
                    shin_len = (np.hypot(l_an[0] - l_kn[0], l_an[1] - l_kn[1]) + np.hypot(r_an[0] - r_kn[0], r_an[1] - r_kn[1])) / 2.0
                    shin_ratio = shin_len / max(0.04, thigh_len)

                    if shin_ratio < CONFIG["shin_ratio_threshold"]:
                        sagittal_knee_deg = float(np.degrees(np.arcsin(np.clip(shin_ratio / 0.85, 0.1, 1.0))) * 1.6)
                        raw_knee = min(angle_kn_2d, sagittal_knee_deg)
                    else:
                        raw_knee = angle_kn_2d

                    chin_buf.append(raw_chin_y)
                    nose_buf.append(nose[1])
                    elbow_buf.append(raw_elbow)
                    knee_buf.append(raw_knee)
                    shoulder_y_buf.append(raw_shoulder_y)

                    chin_y = float(np.median(chin_buf))
                    nose_y = float(np.median(nose_buf))
                    mean_elbow_angle = float(np.median(elbow_buf))
                    mean_knee_angle = float(np.median(knee_buf))
                    shoulder_y = float(np.median(shoulder_y_buf))

                    mean_hip_angle = (calculate_angle(l_sh, l_hip, l_kn) + calculate_angle(r_sh, r_hip, r_kn)) / 2.0

                    curr_sh_w = abs(l_sh[0] - r_sh[0])
                    ankle_center_y = (l_an[1] + r_an[1]) / 2.0
                    curr_body_len = max(0.20, ankle_center_y - shoulder_y)

                    # Калибровка геометрии спортсмена в спокойном висе
                    if mean_elbow_angle >= 170.0 and curr_sh_w > 0.05 and len(shoulder_w_samples) < 50:
                        shoulder_w_samples.append(curr_sh_w)
                        base_shoulder_w = float(np.median(shoulder_w_samples))
                        base_sh_tilt_samples.append(abs(l_sh[1] - r_sh[1]) / max(0.06, base_shoulder_w))
                        base_el_tilt_samples.append(abs(l_el[1] - r_el[1]) / max(0.06, base_shoulder_w))
                        base_el_diff_samples.append(abs(angle_el_l - angle_el_r))
                        base_head_off_samples.append((mouth_x - wrist_x) / max(0.06, base_shoulder_w))
                        base_body_len_samples.append(curr_body_len)

                        base_sh_tilt = float(np.median(base_sh_tilt_samples))
                        base_el_tilt = float(np.median(base_el_tilt_samples))
                        base_el_diff = float(np.median(base_el_diff_samples))
                        base_head_off = float(np.median(base_head_off_samples))
                        base_body_len = float(np.median(base_body_len_samples))

                    # П. 22-24: детекция асимметрии / поочередного сгибания рук
                    curr_sh_tilt = abs(l_sh[1] - r_sh[1]) / max(0.06, base_shoulder_w)
                    curr_el_tilt = abs(l_el[1] - r_el[1]) / max(0.06, base_shoulder_w)
                    curr_head_off = (mouth_x - wrist_x) / max(0.06, base_shoulder_w)
                    elbow_diff = abs(angle_el_l - angle_el_r)

                    net_sh_tilt = max(0.0, curr_sh_tilt - base_sh_tilt)
                    net_el_tilt = max(0.0, curr_el_tilt - base_el_tilt)
                    net_el_diff = max(0.0, elbow_diff - base_el_diff)
                    net_head_shift = abs(curr_head_off - base_head_off)

                    is_asym_frame = (
                        (25.0 < mean_elbow_angle < 155.0)
                        and (
                            net_el_tilt > CONFIG["asym_elbow_tilt_ratio"]
                            or net_sh_tilt > CONFIG["asym_shoulder_tilt_ratio"]
                            or (net_el_diff > CONFIG["asym_elbow_diff_deg"] and net_el_tilt > 0.08)
                            or (net_head_shift > 0.16 and net_el_diff > CONFIG["asym_elbow_diff_deg"] * 0.7)
                        )
                    )

                    # П. 11, 12, 14, 16: положение ног и стоп
                    ankle_dx = l_an[0] - r_an[0]
                    knee_dx = l_kn[0] - r_kn[0]
                    hip_dx = l_hip[0] - r_hip[0]
                    hip_w = max(0.035, abs(hip_dx))
                    ankle_dist = abs(ankle_dx)
                    knee_dist = abs(knee_dx)
                    ankle_dy = abs(l_an[1] - r_an[1])

                    shins_crossed_x = (
                        (ankle_dx * knee_dx < CONFIG["cross_product_threshold"])
                        or (ankle_dx * hip_dx < CONFIG["cross_product_threshold"])
                    )
                    ankles_overlapped = (
                        knee_dist > hip_w * 0.45
                        and ankle_dist < knee_dist * 0.45
                        and ankle_dy > 0.006
                    )
                    legs_crossed = shins_crossed_x or ankles_overlapped

                    feet_apart = ankle_dist > (base_shoulder_w * CONFIG["feet_apart_ratio"])
                    knees_bent = (
                        (mean_knee_angle < CONFIG["knee_bent_threshold"])
                        or (shin_ratio < CONFIG["shin_ratio_threshold"])
                    )

                    # П. 18-20: детекция рывков и махов ногами/туловищем
                    hip_center_x = (l_hip[0] + r_hip[0]) / 2.0
                    ankle_center_x = (l_an[0] + r_an[0]) / 2.0
                    horizontal_swing = (
                        abs(hip_center_x - wrist_x) > (base_shoulder_w * CONFIG["hip_swing_ratio"])
                        or abs(ankle_center_x - hip_center_x) > (base_shoulder_w * CONFIG["ankle_swing_ratio"])
                    )
                    body_foreshortened = (not knees_bent) and (curr_body_len < base_body_len * 0.89)
                    kipping_detected = (
                        (mean_hip_angle < CONFIG["hip_kipping_angle"])
                        or horizontal_swing
                        or body_foreshortened
                    )

                    # П. 26: проверка нахождения кистей на перекладине
                    hands_on_bar = abs(wrist_y - bar_y) < CONFIG["bar_release_dist"]

                    # П. 7, 8, 9: проверка взятия верхней точки
                    chin_above_now = (
                        (chin_y <= bar_y + CONFIG["chin_bar_tolerance"])
                        and (nose_y <= bar_y + CONFIG["nose_bar_margin"])
                        and (mean_elbow_angle <= CONFIG["max_peak_elbow_angle"])
                        and (shoulder_y <= bar_y + 0.12)
                    )

                    if frame_idx <= 5 and state == "WAIT_HANG" and mean_elbow_angle < 65.0 and hands_on_bar:
                        state = "DESCENDING"
                        rep_start_time = 0.0
                        min_elbow_in_rep = mean_elbow_angle
                        min_shoulder_y_in_rep = shoulder_y
                        min_chin_y_in_rep = chin_y
                        chin_above_frames = CONFIG["min_chin_above_frames"] if chin_above_now else 0
                        rep_errors = set()
                        rep_frames = 1
                        asym_frames = knee_bent_frames = legs_crossed_frames = feet_apart_frames = kipping_frames = 0
                        rep_ankle_x = [ankle_center_x]
                        rep_hip_x = [hip_center_x]

                    elif state == "WAIT_HANG":
                        if hands_on_bar:
                            pre_pull_knee_err.append(1 if knees_bent else 0)
                            pre_pull_cross_err.append(1 if legs_crossed else 0)
                            pre_pull_swing_x.append(ankle_center_x)

                        # П. 2: вис на выпрямленных руках
                        if mean_elbow_angle >= elbow_ext_angle and hands_on_bar:
                            if hang_start_time is None:
                                hang_start_time = current_time
                            last_hang_duration = current_time - hang_start_time
                            pre_bend_frames = 0
                        else:
                            # П. 3: вис на полусогнутых руках перед стартом
                            if CONFIG["pull_start_angle"] <= mean_elbow_angle < elbow_ext_angle and hands_on_bar:
                                pre_bend_frames += 1

                            # П. 4: старт тяги
                            if mean_elbow_angle < CONFIG["pull_start_angle"] and hands_on_bar:
                                state = "PULLING"
                                rep_start_time = current_time
                                min_elbow_in_rep = mean_elbow_angle
                                min_shoulder_y_in_rep = shoulder_y
                                min_chin_y_in_rep = chin_y
                                chin_above_frames = 1 if chin_above_now else 0
                                rep_errors = set()
                                rep_frames = 1
                                asym_frames = 1 if is_asym_frame else 0
                                knee_bent_frames = 1 if knees_bent else 0
                                legs_crossed_frames = 1 if legs_crossed else 0
                                feet_apart_frames = 1 if feet_apart else 0
                                kipping_frames = 1 if kipping_detected else 0
                                rep_ankle_x = list(pre_pull_swing_x) + [ankle_center_x]
                                rep_hip_x = [hip_center_x]

                                if pre_bend_frames > int(CONFIG["pre_bend_max_time"] * fps):
                                    rep_errors.add("подтягивание рывками или с махом ногами/туловищем")
                                    rep_errors.add("нарушение ИП: согнутые в локтях руки перед стартом")

                                if sum(pre_pull_knee_err) >= CONFIG["knee_err_min_frames"]:
                                    rep_errors.add("нарушение ИП: согнутые в коленях ноги")
                                if sum(pre_pull_cross_err) >= CONFIG["cross_err_min_frames"]:
                                    rep_errors.add("нарушение ИП: перекрещенные ноги")

                                # П. 1: контроль времени фиксации ИП
                                is_first_rep_early = (len(reps_log) == 0 and rep_start_time < 2.0)
                                if not is_first_rep_early and last_hang_duration < min_hang_sec:
                                    rep_errors.add(f"фиксация ИП менее 1.0 сек ({last_hang_duration:.2f} с)")

                                hang_start_time = None
                                pre_bend_frames = 0

                    elif state in ("PULLING", "DESCENDING"):
                        if not hands_on_bar:
                            # П. 26: срыв / спрыгивание с перекладины до фиксации виса
                            rep_duration = current_time - rep_start_time
                            if rep_duration >= 0.40 and min_elbow_in_rep <= 115.0:
                                if chin_above_frames < CONFIG["min_chin_above_frames"]:
                                    rep_errors.add("подбородок ниже уровня грифа перекладины")
                                rep_errors.add("срыв / спрыгивание с перекладины без фиксации виса")
                                reps_log.append({
                                    "rep_number": len(reps_log) + 1,
                                    "timestamp": f"{format_timestamp(rep_start_time)} – {format_timestamp(current_time)}",
                                    "valid": False,
                                    "status_text": "❌ Не засчитано",
                                    "errors": sorted(rep_errors)
                                })
                            state = "WAIT_HANG"
                            hang_start_time = None
                            last_hang_duration = 0.0
                            pre_bend_frames = 0
                            rep_errors = set()
                            pre_pull_knee_err.clear()
                            pre_pull_cross_err.clear()
                            pre_pull_swing_x.clear()
                        else:
                            rep_frames += 1
                            min_elbow_in_rep = min(min_elbow_in_rep, mean_elbow_angle)
                            min_shoulder_y_in_rep = min(min_shoulder_y_in_rep, shoulder_y)
                            min_chin_y_in_rep = min(min_chin_y_in_rep, chin_y)
                            rep_ankle_x.append(ankle_center_x)
                            rep_hip_x.append(hip_center_x)

                            if chin_above_now:
                                chin_above_frames += 1

                            if is_asym_frame:
                                asym_frames += 1
                            if knees_bent:
                                knee_bent_frames += 1
                            if legs_crossed:
                                legs_crossed_frames += 1
                            if feet_apart:
                                feet_apart_frames += 1
                            if kipping_detected:
                                kipping_frames += 1

                            if state == "PULLING" and (
                                mean_elbow_angle > min_elbow_in_rep + 12.0
                                or shoulder_y > min_shoulder_y_in_rep + 0.025
                            ):
                                state = "DESCENDING"

                            if state == "DESCENDING" and mean_elbow_angle >= (elbow_ext_angle - 4.0):
                                rep_duration = current_time - rep_start_time

                                # П. 5, 6: фильтрация по минимальной длительности и глубине сгибания локтей
                                if (
                                    rep_duration >= CONFIG["min_rep_duration"]
                                    and min_elbow_in_rep <= CONFIG["min_pull_depth_angle"]
                                ):
                                    ankle_swing_span = (max(rep_ankle_x) - min(rep_ankle_x)) if rep_ankle_x else 0.0

                                    # П. 25: длительность поочередного сгибания рук (20% кадров, мин. 5 кадров)
                                    if asym_frames >= max(CONFIG["asym_min_frames"], int(rep_frames * CONFIG["asym_frame_ratio"])):
                                        rep_errors.add("поочередное (неравномерное) сгибание рук")

                                    # П. 13: длительность ошибки согнутых коленей (25% кадров, мин. 5 кадров)
                                    if knee_bent_frames >= max(CONFIG["knee_err_min_frames"], int(rep_frames * CONFIG["knee_err_frame_ratio"])):
                                        rep_errors.add("нарушение ИП: согнутые в коленях ноги")

                                    # П. 15: длительность ошибки скрещивания ног (мин. 5 кадров)
                                    if legs_crossed_frames >= CONFIG["cross_err_min_frames"]:
                                        rep_errors.add("нарушение ИП: перекрещенные ноги")

                                    # П. 17: длительность разведения ног (30% кадров, мин. 5 кадров)
                                    if feet_apart_frames >= max(CONFIG["feet_apart_min_frames"], int(rep_frames * CONFIG["feet_apart_frame_ratio"])):
                                        rep_errors.add("нарушение ИП: ступни не вместе (ноги разведены)")

                                    # П. 21: длительность ошибки маха/рывка (30% кадров, мин. 6 кадров)
                                    if (
                                        kipping_frames >= max(CONFIG["kipping_min_frames"], int(rep_frames * CONFIG["kipping_frame_ratio"]))
                                        or ankle_swing_span > (base_shoulder_w * CONFIG["ankle_swing_ratio"])
                                    ):
                                        rep_errors.add("подтягивание рывками или с махом ногами/туловищем")

                                    # П. 10: число подтвержденных кадров пика
                                    if chin_above_frames < CONFIG["min_chin_above_frames"]:
                                        rep_errors.add("подбородок ниже уровня грифа перекладины")

                                    is_valid = len(rep_errors) == 0
                                    reps_log.append({
                                        "rep_number": len(reps_log) + 1,
                                        "timestamp": f"{format_timestamp(rep_start_time)} – {format_timestamp(current_time)}",
                                        "valid": is_valid,
                                        "status_text": "✅ Засчитано" if is_valid else "❌ Не засчитано",
                                        "errors": sorted(rep_errors) if rep_errors else ["Нет ошибок"]
                                    })

                                state = "WAIT_HANG"
                                hang_start_time = current_time
                                last_hang_duration = 0.0
                                pre_bend_frames = 0
                                rep_errors = set()
                                pre_pull_knee_err.clear()
                                pre_pull_cross_err.clear()
                                pre_pull_swing_x.clear()

                    if writer is not None:
                        draw_skeleton(frame, lm, chin_pt=(mouth_x, chin_y))

                if writer is not None:
                    valid_so_far = sum(1 for r in reps_log if r["valid"])
                    has_active_err = (
                        state != "WAIT_HANG"
                        and (
                            bool(rep_errors)
                            or knee_bent_frames >= CONFIG["knee_err_min_frames"]
                            or legs_crossed_frames >= CONFIG["cross_err_min_frames"]
                            or asym_frames >= CONFIG["asym_min_frames"]
                            or kipping_frames >= CONFIG["kipping_min_frames"]
                        )
                    )
                    draw_hud(
                        frame,
                        valid_reps=valid_so_far,
                        total_attempts=len(reps_log),
                        state=state,
                        hang_sec=last_hang_duration,
                        elbow_deg=mean_elbow_angle,
                        knee_deg=mean_knee_angle,
                        bar_y=bar_y,
                        chin_above_now=chin_above_now,
                        active_errors=has_active_err
                    )
                    writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    final_video_path = None
    if raw_out_path and os.path.exists(raw_out_path):
        final_video_path = convert_to_web_mp4(raw_out_path)

    valid_count = sum(1 for r in reps_log if r["valid"])
    all_unique_errors = sorted({err for r in reps_log if not r["valid"] for err in r["errors"]})
    summary_lines = [
        f"Повторение #{r['rep_number']} [{r['timestamp']}]: {r['status_text']} ({', '.join(r['errors'])})"
        for r in reps_log
    ]

    result["total_attempts"] = len(reps_log)
    result["valid_reps"] = valid_count
    result["bar_y"] = round(bar_y if bar_y is not None else 0.12, 3)
    result["errors"] = all_unique_errors
    result["reps_detail"] = reps_log
    result["comment"] = "\n".join(summary_lines) if summary_lines else "Повторения не зафиксированы."
    result["tracked_video_path"] = final_video_path
    return result
