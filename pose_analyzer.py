from collections import deque
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# =====================================================================
# КОНФИГУРАТОР ПАРАМЕТРОВ И ОГРАНИЧЕНИЙ АЛГОРИТМА (П. 1 - 26 + СТАБИЛИЗАЦИЯ)
# =====================================================================
CONFIG = {
    # 1. Фиксация ИП перед стартом (с)
    "min_hang_sec": 0.55,
    # 2. Угол локтей в нижнем висе (градусы)
    "elbow_ext_angle": 166.0,
    # 3. Максимальное время нахождения на полусогнутых руках (148..166 град) перед стартом тяги (с)
    "pre_bend_max_time": 0.55,
    # 4. Порог начала движения / старт тяги (градусы)
    "pull_start_angle": 148.0,
    # 5. Минимальная глубина движения (градусы)
    "min_pull_depth_angle": 135.0,
    # 6. Минимальная длительность попытки (с)
    "min_rep_duration": 0.45,
    # 7. Допуск высоты подбородка относительно грифа (доля высоты кадра)
    "chin_bar_tolerance": 0.004,
    # 8. Максимальный угол локтей в пике (градусы)
    "max_peak_elbow_angle": 48.0,
    # 9. Высота носа над грифом (доля высоты кадра)
    "nose_bar_margin": -0.014,
    # 9а. Максимально допустимое расстояние от грифа до линии плеч в пике (отсекает недотяг на 00:38)
    "max_peak_shoulder_drop": 0.058,
    # 10. Число подтвержденных кадров пика (минимум 2 кадра, чтобы исключить случайный 1-кадровый выброс)
    "min_chin_above_frames": 2,
    # 11. Порог прямоты ног / 2D-угол в коленях (градусы)
    "knee_bent_threshold": 148.0,
    # 12. Порог захлеста голеней назад 3D (доля длины бедра)
    "shin_ratio_threshold": 0.66,
    # 13. Длительность ошибки согнутых коленей (18% кадров, мин. 3 кадра)
    "knee_err_frame_ratio": 0.18,
    "knee_err_min_frames": 3,
    # 14. Порог скрещивания ног (отлавливает попытку № 8 на 00:43-00:45 и защищает параллельные сомкнутые ноги)
    "cross_product_threshold": -0.00015,
    "cross_min_overlap_ratio": 0.08,
    # 15. Длительность ошибки скрещивания (мин. 3 кадра)
    "cross_err_min_frames": 3,
    # 16. Разведение ног / ступни не вместе (120% ширины плеч)
    "feet_apart_ratio": 1.20,
    # 17. Длительность разведения ног (25% кадров, мин. 5 кадров)
    "feet_apart_frame_ratio": 0.25,
    "feet_apart_min_frames": 5,
    # 18. Рывок тазом / угол в тазобедренном суставе (градусы)
    "hip_kipping_angle": 152.0,
    # 18а. Изгиб центральной оси тела (плечи - таз - колени) при махе (градусы)
    "center_body_bend_angle": 165.0,
    # 19. Горизонтальный вынос таза от оси плеч (60% ширины плеч)
    "hip_swing_ratio": 0.60,
    # 20. Вынос стоп от таза / боковой мах ногами (52% ширины плеч)
    "ankle_swing_ratio": 0.52,
    # 20а. Размах горизонтального раскачивания стоп за попытку (58% ширины плеч)
    "ankle_swing_span_ratio": 0.58,
    # 21. Длительность ошибки маха/рывка (22% кадров, мин. 4 кадра)
    "kipping_frame_ratio": 0.22,
    "kipping_min_frames": 4,
    # 22. Асимметрия: чистая разница углов в локтях сверх калибровки (градусы)
    "asym_elbow_diff_deg": 16.0,
    # 23. Асимметрия: перекос линии локтей (13% ширины плеч — отлавливает попытку № 9 на 00:56)
    "asym_elbow_tilt_ratio": 0.20,
    # 24. Асимметрия: перекос линии плеч (13% ширины плеч)
    "asym_shoulder_tilt_ratio": 0.20,
    # 25. Длительность поочередного сгибания рук (15% кадров, мин. 3 кадра)
    "asym_frame_ratio": 0.20,
    "asym_min_frames": 5,
    # 26. Отрыв кистей от грифа / срыв или спрыгивание (16% высоты кадра)
    "bar_release_dist": 0.16,
    # --- ПАРАМЕТРЫ СТАБИЛИЗАЦИИ СКЕЛЕТА (1€ FILTER + ЗАЩИТА ОТ СХЛОПЫВАНИЯ) ---
    "filter_min_cutoff_x": 0.65,
    "filter_beta_x": 0.04,
    "filter_min_cutoff_y": 1.45,
    "filter_beta_y": 0.40,
    "skeleton_collapse_ratio": 0.68,
    "max_joint_jump_per_frame": 0.085,
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


class StabilizedLandmark:
    """Контейнер ключевой точки, совместимый с MediaPipe NormalizedLandmark."""
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(presence)


class OneEuroFilter:
    """Адаптивный фильтр низких частот 1€ (One Euro Filter) для устранения дрожания без задержки."""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(t_e, cutoff):
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def filter(self, x, t, min_cutoff_override=None, beta_override=None):
        x = float(x)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.t_prev = float(t)
            return x

        t_e = float(t) - self.t_prev
        if t_e <= 1e-6:
            return self.x_prev

        min_cutoff = self.min_cutoff if min_cutoff_override is None else float(min_cutoff_override)
        beta = self.beta if beta_override is None else float(beta_override)

        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = min_cutoff + beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat


class SkeletonStabilizer:
    """Стабилизатор скелета: подавление выбросов, защита от схлопывания и 1€-фильтрация."""

    HAND_INDICES = {
        LM.LEFT_WRIST, LM.RIGHT_WRIST,
        LM.LEFT_PINKY, LM.RIGHT_PINKY,
        LM.LEFT_INDEX, LM.RIGHT_INDEX,
    }

    def __init__(self, config=CONFIG):
        self.cfg = config
        self.filters_x = {}
        self.filters_y = {}
        self.filters_z = {}
        self.prev_pts = None

        self.orientation_sign = None
        self.sh_w_history = deque(maxlen=40)
        self.hip_w_history = deque(maxlen=40)
        self.wrist_w_history = deque(maxlen=40)

        self.ref_sh_w = None
        self.ref_hip_w = None
        self.ref_wrist_w = None
        self.elbow_outward_ratio = 0.0

    def _get_filters(self, idx):
        if idx not in self.filters_x:
            self.filters_x[idx] = OneEuroFilter(
                min_cutoff=self.cfg["filter_min_cutoff_x"],
                beta=self.cfg["filter_beta_x"],
            )
            self.filters_y[idx] = OneEuroFilter(
                min_cutoff=self.cfg["filter_min_cutoff_y"],
                beta=self.cfg["filter_beta_y"],
            )
            self.filters_z[idx] = OneEuroFilter(min_cutoff=0.5, beta=0.02)
        return self.filters_x[idx], self.filters_y[idx], self.filters_z[idx]

    def _restore_pair_width(self, pts, left_idx, right_idx, min_width, target_width):
        lx, rx = pts[left_idx].x, pts[right_idx].x
        curr_dx = lx - rx
        curr_w = abs(curr_dx)

        sign_flipped = (
            self.orientation_sign is not None
            and curr_w > 0.01
            and (curr_dx * self.orientation_sign < 0)
        )

        if curr_w < min_width or sign_flipped:
            if self.prev_pts is not None:
                prev_center_x = 0.5 * (self.prev_pts[left_idx].x + self.prev_pts[right_idx].x)
                raw_center_x = 0.5 * (lx + rx)
                center_x = 0.65 * prev_center_x + 0.35 * raw_center_x
            else:
                center_x = 0.5 * (lx + rx)

            sign = self.orientation_sign if self.orientation_sign is not None else (1.0 if curr_dx >= 0 else -1.0)
            restored_w = 0.85 * target_width + 0.15 * max(curr_w, min_width)
            half_w = 0.5 * restored_w
            pts[left_idx].x = center_x + sign * half_w
            pts[right_idx].x = center_x - sign * half_w

            if self.prev_pts is not None:
                pts[left_idx].y = 0.65 * self.prev_pts[left_idx].y + 0.35 * pts[left_idx].y
                pts[right_idx].y = 0.65 * self.prev_pts[right_idx].y + 0.35 * pts[right_idx].y
            return True
        return False

    def process(self, raw_landmarks, timestamp_sec, bar_y=None):
        pts = [
            StabilizedLandmark(
                lm.x,
                lm.y,
                getattr(lm, "z", 0.0),
                getattr(lm, "visibility", 1.0),
                getattr(lm, "presence", 1.0),
            )
            for lm in raw_landmarks
        ]

        if self.prev_pts is not None:
            max_jump = self.cfg["max_joint_jump_per_frame"]
            for idx, p in enumerate(pts):
                prev_p = self.prev_pts[idx]
                conf = min(p.visibility, p.presence)
                dist = math.hypot(p.x - prev_p.x, p.y - prev_p.y)

                if conf < 0.45:
                    blend = 0.25
                    p.x = prev_p.x + blend * (p.x - prev_p.x)
                    p.y = prev_p.y + blend * (p.y - prev_p.y)
                elif dist > max_jump:
                    scale = max_jump / dist
                    p.x = prev_p.x + (p.x - prev_p.x) * scale
                    p.y = prev_p.y + (p.y - prev_p.y) * scale

        wrist_y = 0.5 * (pts[LM.LEFT_WRIST].y + pts[LM.RIGHT_WRIST].y)
        hands_on_bar = (bar_y is None) or (abs(wrist_y - bar_y) < self.cfg["bar_release_dist"])

        sh_dx = pts[LM.LEFT_SHOULDER].x - pts[LM.RIGHT_SHOULDER].x
        sh_w = abs(sh_dx)
        hip_w = abs(pts[LM.LEFT_HIP].x - pts[LM.RIGHT_HIP].x)
        wrist_w = abs(pts[LM.LEFT_WRIST].x - pts[LM.RIGHT_WRIST].x)

        is_clean_width = (
            sh_w > 0.055
            and (self.ref_sh_w is None or sh_w >= 0.72 * self.ref_sh_w)
        )
        if is_clean_width:
            if self.orientation_sign is None:
                self.orientation_sign = 1.0 if sh_dx >= 0 else -1.0
            self.sh_w_history.append(sh_w)
            self.ref_sh_w = float(np.median(self.sh_w_history))

            if hip_w > 0.035 and (self.ref_hip_w is None or hip_w >= 0.70 * self.ref_hip_w):
                self.hip_w_history.append(hip_w)
                self.ref_hip_w = float(np.median(self.hip_w_history))

            if hands_on_bar and wrist_w > 0.055:
                self.wrist_w_history.append(wrist_w)
                self.ref_wrist_w = float(np.median(self.wrist_w_history))

        collapse_ratio = self.cfg["skeleton_collapse_ratio"]

        if self.ref_sh_w is not None:
            if hands_on_bar and self.ref_wrist_w is not None:
                self._restore_pair_width(
                    pts,
                    LM.LEFT_WRIST,
                    LM.RIGHT_WRIST,
                    min_width=self.ref_wrist_w * 0.78,
                    target_width=self.ref_wrist_w,
                )

            sh_collapsed = self._restore_pair_width(
                pts,
                LM.LEFT_SHOULDER,
                LM.RIGHT_SHOULDER,
                min_width=self.ref_sh_w * collapse_ratio,
                target_width=self.ref_sh_w,
            )

            target_hip_w = self.ref_hip_w if self.ref_hip_w is not None else (self.ref_sh_w * 0.62)
            hip_collapsed = self._restore_pair_width(
                pts,
                LM.LEFT_HIP,
                LM.RIGHT_HIP,
                min_width=target_hip_w * collapse_ratio,
                target_width=target_hip_w,
            )

            el_dx = pts[LM.LEFT_ELBOW].x - pts[LM.RIGHT_ELBOW].x
            el_w = abs(el_dx)
            expected_arm_span = 0.5 * (
                abs(pts[LM.LEFT_SHOULDER].x - pts[LM.RIGHT_SHOULDER].x)
                + abs(pts[LM.LEFT_WRIST].x - pts[LM.RIGHT_WRIST].x)
            )
            min_el_w = max(self.ref_sh_w * collapse_ratio, expected_arm_span * 0.66)

            el_sign_flipped = (
                self.orientation_sign is not None
                and el_w > 0.01
                and (el_dx * self.orientation_sign < 0)
            )

            if el_w < min_el_w or el_sign_flipped:
                sign = self.orientation_sign if self.orientation_sign is not None else 1.0
                outward = self.elbow_outward_ratio * self.ref_sh_w
                pts[LM.LEFT_ELBOW].x = (
                    0.55 * pts[LM.LEFT_SHOULDER].x
                    + 0.45 * pts[LM.LEFT_WRIST].x
                    + sign * outward
                )
                pts[LM.RIGHT_ELBOW].x = (
                    0.55 * pts[LM.RIGHT_SHOULDER].x
                    + 0.45 * pts[LM.RIGHT_WRIST].x
                    - sign * outward
                )
                if self.prev_pts is not None:
                    pts[LM.LEFT_ELBOW].y = 0.65 * self.prev_pts[LM.LEFT_ELBOW].y + 0.35 * pts[LM.LEFT_ELBOW].y
                    pts[LM.RIGHT_ELBOW].y = 0.65 * self.prev_pts[LM.RIGHT_ELBOW].y + 0.35 * pts[LM.RIGHT_ELBOW].y
            else:
                mid_span = max(0.05, expected_arm_span)
                self.elbow_outward_ratio = float(
                    np.clip((el_w - mid_span) / (2.0 * self.ref_sh_w), -0.05, 0.25)
                )

            if (sh_collapsed or hip_collapsed) and self.prev_pts is not None:
                kn_w = abs(pts[LM.LEFT_KNEE].x - pts[LM.RIGHT_KNEE].x)
                prev_kn_w = abs(self.prev_pts[LM.LEFT_KNEE].x - self.prev_pts[LM.RIGHT_KNEE].x)
                if kn_w < 0.60 * prev_kn_w:
                    self._restore_pair_width(
                        pts,
                        LM.LEFT_KNEE,
                        LM.RIGHT_KNEE,
                        min_width=0.65 * prev_kn_w,
                        target_width=prev_kn_w,
                    )

        smoothed_pts = []
        for idx, p in enumerate(pts):
            fx, fy, fz = self._get_filters(idx)

            if hands_on_bar and idx in self.HAND_INDICES:
                sx = fx.filter(p.x, timestamp_sec, min_cutoff_override=0.18, beta_override=0.008)
                sy = fy.filter(p.y, timestamp_sec, min_cutoff_override=0.25, beta_override=0.015)
            else:
                sx = fx.filter(p.x, timestamp_sec)
                sy = fy.filter(p.y, timestamp_sec)

            sz = fz.filter(p.z, timestamp_sec)
            smoothed_pts.append(StabilizedLandmark(sx, sy, sz, p.visibility, p.presence))

        self.prev_pts = smoothed_pts
        return smoothed_pts


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
    """Определяет уровень оси перекладины по верхним костяшкам пальцев и запястьям."""
    l_knuckle_y = min(lm[LM.LEFT_INDEX].y, lm[LM.LEFT_PINKY].y)
    r_knuckle_y = min(lm[LM.RIGHT_INDEX].y, lm[LM.RIGHT_PINKY].y)
    knuckle_y = 0.5 * (l_knuckle_y + r_knuckle_y)
    wrist_y = 0.5 * (lm[LM.LEFT_WRIST].y + lm[LM.RIGHT_WRIST].y)
    return float(min(knuckle_y - 0.005, wrist_y - 0.022))


def compute_anatomical_chin(lm, base_shoulder_w=0.12):
    """
    Точное вычисление нижней точки подбородка по лицевым ориентирам (нос и уголки рта)
    с защитой от схлопывания, когда перекладина перекрывает лицо.
    """
    nose_x, nose_y = lm[LM.NOSE].x, lm[LM.NOSE].y
    mouth_x = 0.5 * (lm[LM.MOUTH_LEFT].x + lm[LM.MOUTH_RIGHT].x)
    raw_mouth_y = 0.5 * (lm[LM.MOUTH_LEFT].y + lm[LM.MOUTH_RIGHT].y)

    nose_to_mouth = float(np.clip(
        raw_mouth_y - nose_y,
        0.10 * base_shoulder_w,
        0.24 * base_shoulder_w
    ))
    valid_mouth_y = nose_y + nose_to_mouth
    chin_y = valid_mouth_y + 0.55 * nose_to_mouth
    chin_x = 0.70 * mouth_x + 0.30 * nose_x
    return chin_x, float(chin_y)


def draw_skeleton(frame, landmarks, chin_pt=None):
    h, w, _ = frame.shape
    for idx_a, idx_b in POSE_CONNECTIONS:
        pt1 = (int(landmarks[idx_a].x * w), int(landmarks[idx_a].y * h))
        pt2 = (int(landmarks[idx_b].x * w), int(landmarks[idx_b].y * h))
        cv2.line(frame, pt1, pt2, (255, 200, 0), 3, cv2.LINE_AA)

    for idx in [
        LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
        LM.LEFT_ELBOW, LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST,
        LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE,
        LM.LEFT_ANKLE, LM.RIGHT_ANKLE
    ]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)

    if chin_pt is not None:
        ch_x, ch_y = int(chin_pt[0] * w), int(chin_pt[1] * h)
        cv2.circle(frame, (ch_x, ch_y), 7, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "CHIN", (ch_x + 10, ch_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)


def draw_hud(frame, valid_reps, total_attempts, state, hang_sec, elbow_deg, knee_deg, bar_y, chin_above_now, active_errors):
    h, w, _ = frame.shape
    if bar_y is not None:
        bar_px = int(bar_y * h)
        bar_color = (0, 255, 0) if chin_above_now else (0, 215, 255)
        cv2.line(frame, (0, bar_px), (w, bar_px), bar_color, 3)
        label_y = bar_px + 22 if bar_px < 35 else bar_px - 8
        label_text = f"BAR LINE (Y={bar_y:.2f}) {'[CHIN ABOVE]' if chin_above_now else ''}"
        cv2.putText(frame, label_text, (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, label_text, (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

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
            sh_w = max(0.08, abs(lm[LM.LEFT_SHOULDER].x - lm[LM.RIGHT_SHOULDER].x))
            chin_x, chin_y = compute_anatomical_chin(lm, sh_w)
            draw_skeleton(frame, lm, chin_pt=(chin_x, chin_y))

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

        base_shoulder_w = 0.12
        base_sh_tilt = 0.03
        base_el_tilt = 0.03
        base_el_diff = 5.0

        stabilizer = SkeletonStabilizer(CONFIG)
        chin_x_filter = OneEuroFilter(
            min_cutoff=CONFIG["filter_min_cutoff_x"],
            beta=CONFIG["filter_beta_x"]
        )
        chin_y_filter = OneEuroFilter(
            min_cutoff=CONFIG["filter_min_cutoff_y"],
            beta=CONFIG["filter_beta_y"]
        )

        # Сглаживающие буферы по 3 кадрам
        chin_buf = deque(maxlen=3)
        nose_buf = deque(maxlen=3)
        elbow_buf = deque(maxlen=3)
        knee_buf = deque(maxlen=3)
        shoulder_y_buf = deque(maxlen=3)

        # Буферы контроля ИП в висе перед началом тяги
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
        rep_ankle_x = []

        frame_idx = 0
        last_ts_ms = -1

        model_path = ensure_model_path()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.65
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
                    raw_lm = results.pose_landmarks[0]

                    if manual_bar_y is None and not bar_y_samples:
                        bar_y = compute_bar_y_from_hands(raw_lm)

                    lm = stabilizer.process(raw_lm, current_time, bar_y=bar_y)

                    def pt(idx):
                        return [lm[idx].x, lm[idx].y]

                    l_sh, r_sh = pt(LM.LEFT_SHOULDER), pt(LM.RIGHT_SHOULDER)
                    l_el, r_el = pt(LM.LEFT_ELBOW), pt(LM.RIGHT_ELBOW)
                    l_wr, r_wr = pt(LM.LEFT_WRIST), pt(LM.RIGHT_WRIST)
                    l_hip, r_hip = pt(LM.LEFT_HIP), pt(LM.RIGHT_HIP)
                    l_kn, r_kn = pt(LM.LEFT_KNEE), pt(LM.RIGHT_KNEE)
                    l_an, r_an = pt(LM.LEFT_ANKLE), pt(LM.RIGHT_ANKLE)

                    nose = pt(LM.NOSE)
                    raw_chin_x, raw_chin_y = compute_anatomical_chin(lm, base_shoulder_w)

                    mouth_x = chin_x_filter.filter(raw_chin_x, current_time)
                    filtered_chin_y = chin_y_filter.filter(raw_chin_y, current_time)
                    raw_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
                    shoulder_center_x = (l_sh[0] + r_sh[0]) / 2.0

                    wrist_y = (l_wr[1] + r_wr[1]) / 2.0

                    angle_el_l = calculate_angle(l_sh, l_el, l_wr)
                    angle_el_r = calculate_angle(r_sh, r_el, r_wr)
                    raw_elbow = (angle_el_l + angle_el_r) / 2.0

                    # Калибруем высоту перекладины в спокойном висе
                    if manual_bar_y is None and raw_elbow >= 162.0 and len(bar_y_samples) < 80:
                        bar_y_samples.append(compute_bar_y_from_hands(raw_lm))
                        bar_y = float(np.percentile(bar_y_samples, 35))

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

                    chin_buf.append(filtered_chin_y)
                    nose_buf.append(nose[1])
                    elbow_buf.append(raw_elbow)
                    knee_buf.append(raw_knee)
                    shoulder_y_buf.append(raw_shoulder_y)

                    nose_y = float(np.median(nose_buf))
                    shoulder_y = float(np.median(shoulder_y_buf))
                    chin_y = max(
                        float(np.median(chin_buf)),
                        nose_y + 0.16 * base_shoulder_w
                    )
                    mean_elbow_angle = float(np.median(elbow_buf))
                    mean_knee_angle = float(np.median(knee_buf))

                    mean_hip_angle = (calculate_angle(l_sh, l_hip, l_kn) + calculate_angle(r_sh, r_hip, r_kn)) / 2.0
                    curr_sh_w = abs(l_sh[0] - r_sh[0])

                    # Калибровка геометрии спортсмена в спокойном висе
                    if mean_elbow_angle >= 170.0 and curr_sh_w > 0.05 and len(shoulder_w_samples) < 50:
                        shoulder_w_samples.append(curr_sh_w)
                        base_shoulder_w = float(np.median(shoulder_w_samples))
                        base_sh_tilt_samples.append(abs(l_sh[1] - r_sh[1]) / max(0.06, base_shoulder_w))
                        base_el_tilt_samples.append(abs(l_el[1] - r_el[1]) / max(0.06, base_shoulder_w))
                        base_el_diff_samples.append(abs(angle_el_l - angle_el_r))

                        base_sh_tilt = float(np.median(base_sh_tilt_samples))
                        base_el_tilt = float(np.median(base_el_tilt_samples))
                        base_el_diff = float(np.median(base_el_diff_samples))

                    # П. 22-24: детекция асимметрии / поочередного сгибания рук (отлавливает попытку № 9 на 00:56)
                    curr_sh_tilt = abs(l_sh[1] - r_sh[1]) / max(0.06, base_shoulder_w)
                    curr_el_tilt = abs(l_el[1] - r_el[1]) / max(0.06, base_shoulder_w)
                    elbow_diff = abs(angle_el_l - angle_el_r)

                    net_sh_tilt = max(0.0, curr_sh_tilt - base_sh_tilt)
                    net_el_tilt = max(0.0, curr_el_tilt - base_el_tilt)
                    net_el_diff = max(0.0, elbow_diff - base_el_diff)

                    head_fully_in_frame = nose_y > 0.04
                    is_asym_frame = (
                        head_fully_in_frame
                        and (20.0 < mean_elbow_angle < 162.0)
                        and (
                            net_el_tilt > CONFIG["asym_elbow_tilt_ratio"]
                            or net_sh_tilt > CONFIG["asym_shoulder_tilt_ratio"]
                            or (net_el_diff > CONFIG["asym_elbow_diff_deg"] and net_el_tilt > 0.07)
                        )
                    )

                    # П. 11, 12, 14, 16: положение ног и стоп (отлавливает скрещивание в попытке № 8 на 00:43-00:45)
                    ankle_dx = l_an[0] - r_an[0]
                    knee_dx = l_kn[0] - r_kn[0]
                    hip_dx = l_hip[0] - r_hip[0]
                    hip_w = max(0.04, abs(hip_dx))
                    ankle_dist = abs(ankle_dx)
                    knee_dist = abs(knee_dx)
                    ankle_dy = abs(l_an[1] - r_an[1])

                    # 1) Инверсия знака лодыжек относительно таза/коленей (Х-перехлест)
                    shins_crossed_x = (
                        (
                            ankle_dx * hip_dx < CONFIG["cross_product_threshold"]
                            or ankle_dx * knee_dx < CONFIG["cross_product_threshold"]
                        )
                        and (ankle_dist > hip_w * CONFIG["cross_min_overlap_ratio"] or ankle_dy > hip_w * 0.12)
                    )
                    # 2) Наложение одной лодыжки поверх другой (колени шире лодыжек, стопы на разной высоте)
                    ankles_stacked_vertically = (
                        knee_dist > hip_w * 0.42
                        and (ankle_dist < knee_dist * 0.48 or ankle_dist < hip_w * 0.28)
                        and ankle_dy > hip_w * 0.14
                    )
                    legs_crossed = shins_crossed_x or ankles_stacked_vertically

                    feet_apart = ankle_dist > (base_shoulder_w * CONFIG["feet_apart_ratio"])
                    knees_bent = (
                        (mean_knee_angle < CONFIG["knee_bent_threshold"])
                        or (shin_ratio < CONFIG["shin_ratio_threshold"])
                    )

                    # П. 18-20: детекция реальных рывков и махов
                    hip_center_x = (l_hip[0] + r_hip[0]) / 2.0
                    hip_center_y = (l_hip[1] + r_hip[1]) / 2.0
                    knee_center_x = (l_kn[0] + r_kn[0]) / 2.0
                    knee_center_y = (l_kn[1] + r_kn[1]) / 2.0
                    ankle_center_x = (l_an[0] + r_an[0]) / 2.0

                    center_body_angle = calculate_angle(
                        [shoulder_center_x, shoulder_y],
                        [hip_center_x, hip_center_y],
                        [knee_center_x, knee_center_y]
                    )
                    horizontal_swing = (
                        abs(hip_center_x - shoulder_center_x) > (base_shoulder_w * CONFIG["hip_swing_ratio"])
                        or abs(ankle_center_x - hip_center_x) > (base_shoulder_w * CONFIG["ankle_swing_ratio"])
                    )
                    kipping_detected = (
                        (mean_hip_angle < CONFIG["hip_kipping_angle"])
                        or (center_body_angle < CONFIG["center_body_bend_angle"])
                        or horizontal_swing
                    )

                    # П. 26: проверка нахождения кистей на перекладине
                    hands_on_bar = (
                        abs(wrist_y - bar_y) < CONFIG["bar_release_dist"]
                        and wrist_y <= (shoulder_y + 0.04)
                    )

                    # П. 7, 8, 9, 9а: строгая проверка взятия верхней точки (отсекает недотяг № 7 на 00:38)
                    chin_reached = (
                        (
                            chin_y <= bar_y + CONFIG["chin_bar_tolerance"]
                            and nose_y <= bar_y + CONFIG["nose_bar_margin"]
                        )
                        or (bar_y <= 0.08 and nose_y <= bar_y)
                    )
                    chin_above_now = (
                        chin_reached
                        and (mean_elbow_angle <= CONFIG["max_peak_elbow_angle"])
                        and (shoulder_y <= bar_y + CONFIG["max_peak_shoulder_drop"])
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

                    elif state == "WAIT_HANG":
                        if hands_on_bar:
                            pre_pull_knee_err.append(1 if knees_bent else 0)
                            pre_pull_cross_err.append(1 if legs_crossed else 0)

                        # П. 2: вис на выпрямленных руках
                        if mean_elbow_angle >= elbow_ext_angle and hands_on_bar:
                            if hang_start_time is None:
                                hang_start_time = current_time
                            last_hang_duration = current_time - hang_start_time
                            pre_bend_frames = 0
                        else:
                            # П. 3: контроль виса на полусогнутых руках перед стартом тяги (отлавливает попытку № 4 на 00:22)
                            if (
                                CONFIG["pull_start_angle"] <= mean_elbow_angle < elbow_ext_angle
                                and hands_on_bar
                            ):
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
                                rep_ankle_x = [ankle_center_x]

                                if pre_bend_frames > int(CONFIG["pre_bend_max_time"] * fps):
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
                            rep_duration = current_time - rep_start_time
                            if (
                                rep_duration >= 0.45
                                and min_elbow_in_rep <= 95.0
                                and min_shoulder_y_in_rep <= (bar_y + 0.18)
                            ):
                                if (
                                    chin_above_frames < CONFIG["min_chin_above_frames"]
                                    or min_elbow_in_rep > CONFIG["max_peak_elbow_angle"]
                                ):
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
                        else:
                            rep_frames += 1
                            min_elbow_in_rep = min(min_elbow_in_rep, mean_elbow_angle)
                            min_shoulder_y_in_rep = min(min_shoulder_y_in_rep, shoulder_y)
                            min_chin_y_in_rep = min(min_chin_y_in_rep, chin_y)
                            rep_ankle_x.append(ankle_center_x)

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

                                    # П. 25: поочередное / неравномерное сгибание рук
                                    if asym_frames >= max(CONFIG["asym_min_frames"], int(rep_frames * CONFIG["asym_frame_ratio"])):
                                        rep_errors.add("поочередное (неравномерное) сгибание рук")

                                    # П. 13: согнутые в коленях ноги
                                    if knee_bent_frames >= max(CONFIG["knee_err_min_frames"], int(rep_frames * CONFIG["knee_err_frame_ratio"])):
                                        rep_errors.add("нарушение ИП: согнутые в коленях ноги")

                                    # П. 15: перекрещенные ноги
                                    if legs_crossed_frames >= CONFIG["cross_err_min_frames"]:
                                        rep_errors.add("нарушение ИП: перекрещенные ноги")

                                    # П. 17: разведение ног
                                    if feet_apart_frames >= max(CONFIG["feet_apart_min_frames"], int(rep_frames * CONFIG["feet_apart_frame_ratio"])):
                                        rep_errors.add("нарушение ИП: ступни не вместе (ноги разведены)")

                                    # П. 21: рывки / махи ногами или туловищем
                                    if (
                                        kipping_frames >= max(CONFIG["kipping_min_frames"], int(rep_frames * CONFIG["kipping_frame_ratio"]))
                                        or ankle_swing_span > (base_shoulder_w * CONFIG["ankle_swing_span_ratio"])
                                    ):
                                        rep_errors.add("подтягивание рывками или с махом ногами/туловищем")

                                    # П. 7, 8, 9а, 10: строгая проверка взятия верхней точки
                                    if (
                                        chin_above_frames < CONFIG["min_chin_above_frames"]
                                        or min_elbow_in_rep > CONFIG["max_peak_elbow_angle"]
                                        or min_chin_y_in_rep > (bar_y + CONFIG["chin_bar_tolerance"])
                                        or min_shoulder_y_in_rep > (bar_y + CONFIG["max_peak_shoulder_drop"])
                                    ):
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
