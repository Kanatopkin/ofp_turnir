from collections import deque
import json
import math
import os
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# Безопасный импорт SDK для Gemini API (поддерживает новый google-genai и классический google-generativeai)
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_SDK = "google-genai"
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GEMINI_SDK = "legacy"
    except ImportError:
        GEMINI_SDK = None

# ПУНКТ 2 и 3: Легковесная модель YOLOv8n-Pose (Nano) + поддержка OpenVINO / ONNX
YOLO_MODEL_NAME = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
GEMINI_MODEL_NAME = os.getenv("GEMINI_REFEREE_MODEL", "gemini-3.5-flash")
_MODEL_INSTANCE = None

# =====================================================================
# КОНФИГУРАТОР ПАРАМЕТРОВ И ОГРАНИЧЕНИЙ АЛГОРИТМА (П. 1 - 26 + УСКОРЕНИЕ + ИИ-АРБИТР)
# =====================================================================
CONFIG = {
    # --- ПАРАМЕТРЫ УСКОРЕНИЯ ИНФЕРЕНСА (П. 1 - 3) ---
    "target_eval_fps": 15.0,            # Нормализация частоты анализа (15 к/с без потери точности)
    "yolo_imgsz": 480,                  # Оптимальное разрешение для YOLO Pose (ускорение на 35%)
    # 1. Фиксация ИП перед стартом (с)
    "min_hang_sec": 0.55,
    # 2. Угол локтей в нижнем висе (градусы)
    "elbow_ext_angle": 166.0,
    # 3. Максимальное время нахождения на полусогнутых руках (148..166 град) перед стартом тяги (с)
    "pre_bend_max_time": 0.65,
    # 4. Порог начала движения / старт тяги (градусы)
    "pull_start_angle": 148.0,
    # 5. Минимальная глубина движения (градусы)
    "min_pull_depth_angle": 135.0,
    # 6. Минимальная длительность попытки (с)
    "min_rep_duration": 0.45,
    # 7. Допуск высоты подбородка относительно грифа (доля высоты кадра)
    "chin_bar_tolerance": 0.006,
    # 8. Максимальный угол локтей в пике (градусы)
    "max_peak_elbow_angle": 48.0,
    # 9. Высота носа над грифом (доля высоты кадра)
    "nose_bar_margin": -0.010,
    # 9а. Максимально допустимое расстояние от грифа до линии плеч в пике
    "max_peak_shoulder_drop": 0.052,
    # 10. Число подтвержденных кадров пика
    "min_chin_above_frames": 1,
    # 11. Порог прямоты ног / 2D-угол в коленях (градусы)
    "knee_bent_threshold": 145.0,
    # 12. Порог захлеста голеней назад 3D (доля длины бедра)
    "shin_ratio_threshold": 0.66,
    # 13. Длительность ошибки согнутых коленей (20% кадров, эталон для 30 к/с — 4 кадра)
    "knee_err_frame_ratio": 0.20,
    "knee_err_min_frames": 4,
    # 14. Порог скрещивания ног (исключает шум при сомкнутых ступнях)
    "cross_product_threshold": -0.0002,
    "cross_min_overlap_ratio": 0.14,
    # 15. Длительность ошибки скрещивания (эталон для 30 к/с — 3 кадра)
    "cross_err_min_frames": 3,
    # 16. Разведение ног / ступни не вместе (120% ширины плеч)
    "feet_apart_ratio": 1.20,
    # 17. Длительность разведения ног (25% кадров, эталон для 30 к/с — 5 кадров)
    "feet_apart_frame_ratio": 0.25,
    "feet_apart_min_frames": 5,
    # 18. Рывок тазом / угол в тазобедренном суставе (градусы)
    "hip_kipping_angle": 151.0,
    # 18а. Изгиб центральной оси тела (плечи - таз - колени) при махе (градусы)
    "center_body_bend_angle": 165.0,
    # 19. Горизонтальный вынос таза от оси плеч (60% ширины плеч)
    "hip_swing_ratio": 0.60,
    # 20. Вынос стоп от таза / боковой мах ногами (52% ширины плеч)
    "ankle_swing_ratio": 0.52,
    # 20а. Размах горизонтального раскачивания стоп за попытку (58% ширины плеч)
    "ankle_swing_span_ratio": 0.58,
    # 21. Длительность ошибки маха/рывка (25% кадров, эталон для 30 к/с — 5 кадров)
    "kipping_frame_ratio": 0.25,
    "kipping_min_frames": 5,
    # 22. Асимметрия: чистая разница углов в локтях сверх калибровки (градусы)
    "asym_elbow_diff_deg": 18.0,
    # 23. Асимметрия: перекос линии локтей (14% ширины плеч)
    "asym_elbow_tilt_ratio": 0.14,
    # 24. Асимметрия: перекос линии плеч (14% ширины плеч)
    "asym_shoulder_tilt_ratio": 0.14,
    # 25. Длительность поочередного сгибания рук (16% кадров, эталон для 30 к/с — 4 кадра)
    "asym_frame_ratio": 0.16,
    "asym_min_frames": 4,
    # 26. Отрыв кистей от грифа / срыв или спрыгивание (16% высоты кадра)
    "bar_release_dist": 0.16,
    # --- ПАРАМЕТРЫ ДИНАМИЧЕСКОГО ОБНОВЛЕНИЯ ЛИНИИ ПЕРЕКЛАДИНЫ НА КАЖДОЕ ПОДТЯГИВАНИЕ ---
    "bar_hang_window_frames": 10,       # Окно калибровки грифа перед началом каждого нового подтягивания
    "bar_rep_window_frames": 12,        # Окно адаптивного уточнения грифа внутри текущего подтягивания
    "bar_max_Rep_drift": 0.045,         # Максимально допустимое смещение грифа за кадр внутри попытки
    # --- ПАРАМЕТРЫ СТАБИЛИЗАЦИИ СКЕЛЕТА (1€ FILTER + ЗАЩИТА ОТ СХЛОПЫВАНИЯ) ---
    "filter_min_cutoff_x": 0.65,
    "filter_beta_x": 0.04,
    "filter_min_cutoff_y": 1.35,
    "filter_beta_y": 0.35,
    "skeleton_collapse_ratio": 0.68,
    "max_joint_jump_per_frame": 0.14,   # Адаптировано под шаг 15 FPS
    # --- ПАРАМЕТРЫ УЧАЩЕННОЙ ВЕРИФИКАЦИИ ЧЕРЕЗ GEMINI API И ИНДИКАТОРА ---
    "enable_gemini_verifier": True,
    "dispute_chin_margin": 0.050,       # Зона сомнения по высоте подбородка у грифа (±5% высоты кадра)
    "dispute_frame_margin": 4,          # Запас кадров вокруг порога ошибки для вызова ИИ-арбитра
    "ai_banner_display_sec": 3.0,       # Длительность свечения баннера ИИ-арбитра на видео (с)
}


class LM:
    """Индексы ключевых точек стандарта COCO 17 (YOLOv8-Pose)."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


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
    """Унифицированный контейнер ключевой точки."""
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(presence)


def _resolve_gemini_api_key():
    """Ищет ключ GEMINI_API_KEY в переменных окружения или в Streamlit Secrets."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            return st.secrets["gemini"]["api_key"]
    except Exception:
        pass
    return None


class GeminiDisputeVerifier:
    """
    Верифицирующий модуль (ИИ-арбитр) на базе Gemini 3.5 Flash для проверки спорных попыток
    с повышенным вниманием к подбородку, асимметрии рук и скрещиванию ног.
    """

    def __init__(self, enabled=True):
        self.api_key = _resolve_gemini_api_key()
        self.enabled = bool(enabled and self.api_key and GEMINI_SDK is not None)
        self.client = None
        if self.enabled:
            try:
                if GEMINI_SDK == "google-genai":
                    self.client = genai.Client(api_key=self.api_key)
                elif GEMINI_SDK == "legacy":
                    genai_legacy.configure(api_key=self.api_key)
                    self.client = genai_legacy.GenerativeModel(GEMINI_MODEL_NAME)
            except Exception:
                self.enabled = False

    @staticmethod
    def get_dispute_reasons(
        rep_errors,
        min_chin_y,
        bar_y,
        min_elbow_deg,
        asym_frames,
        asym_thresh,
        knee_frames,
        knee_thresh,
        cross_frames,
        cross_thresh,
        kip_frames,
        kip_thresh,
    ):
        """Возвращает список причин, по которым попытка требует вмешательства ИИ-арбитра."""
        reasons = []
        margin = CONFIG["dispute_frame_margin"]

        # 1. Контроль подбородка относительно обновленной линии перекладины
        if (
            abs(min_chin_y - bar_y) <= CONFIG["dispute_chin_margin"]
            or "подбородок ниже уровня грифа перекладины" in rep_errors
            or (min_elbow_deg <= 68.0 and min_elbow_deg > 28.0)
        ):
            reasons.append("CHIN")

        # 2. Контроль асимметрии рук
        if asym_frames >= 1 or abs(asym_frames - asym_thresh) <= margin:
            reasons.append("ASYM_ARMS")

        # 3. Контроль скрещивания ног
        if cross_frames >= 1 or abs(cross_frames - cross_thresh) <= margin:
            reasons.append("CROSS_LEGS")

        # 4. Контроль коленей и махов
        if knee_frames >= 1 or abs(knee_frames - knee_thresh) <= margin:
            reasons.append("KNEES")
        if kip_frames >= 1 or abs(kip_frames - kip_thresh) <= margin:
            reasons.append("KIPPING")

        return reasons

    @classmethod
    def is_rep_disputed(cls, **kwargs):
        return len(cls.get_dispute_reasons(**kwargs)) > 0

    @staticmethod
    def build_storyboard_jpeg(start_frame, mid_frame, peak_frame, bar_y):
        """Склеивает 3 ключевых кадра (ИП, Середина тяги, Пик) в одно изображение с разметкой линии грифа."""
        frames = [f for f in (start_frame, mid_frame, peak_frame) if f is not None]
        if not frames:
            return None

        titles = ["1. HANG (START)", "2. MID PULL", "3. PEAK (TOP)"]
        panels = []
        target_h = 360

        for idx, img in enumerate(frames):
            h, w = img.shape[:2]
            scale = target_h / max(1, h)
            new_w = max(2, int(w * scale))
            resized = cv2.resize(img, (new_w, target_h))

            if bar_y is not None:
                by = int(bar_y * target_h)
                cv2.line(resized, (0, by), (new_w, by), (0, 255, 255), 2)

            title = titles[idx] if idx < len(titles) else f"FRAME {idx + 1}"
            cv2.rectangle(resized, (0, 0), (new_w, 28), (15, 15, 15), -1)
            cv2.putText(resized, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            panels.append(resized)

        collage = np.hstack(panels)
        ok, buf = cv2.imencode(".jpg", collage, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else None

    def verify_repetition(self, start_frame, mid_frame, peak_frame, bar_y, fsm_errors, min_elbow_deg):
        """
        Отправляет раскадровку спорной попытки в Gemini 3.5 Flash и возвращает:
        (скорректированный набор ошибок, краткий вердикт ИИ-арбитра).
        """
        if not self.enabled:
            return set(fsm_errors), None

        jpeg_bytes = self.build_storyboard_jpeg(start_frame, mid_frame, peak_frame, bar_y)
        if not jpeg_bytes:
            return set(fsm_errors), None

        prompt = (
            "Ты — главный судья соревнований ГТО/ОФП по подтягиваниям из виса на высокой перекладине.\n"
            "На изображении показаны 3 фазы одной спорной попытки: "
            "1. HANG (исходное положение виса), 2. MID PULL (середина тяги), 3. PEAK (верхняя точка).\n"
            "Желтая горизонтальная линия — индивидуально откалиброванный уровень грифа перекладины для этого повторения.\n"
            f"Предварительные замечания локального датчика: {list(fsm_errors) if fsm_errors else ['Нет ошибок']} "
            f"(минимальный угол локтей в пике: {min_elbow_deg:.0f} град).\n\n"
            "Особое внимание удели 3 главным триггерам: ПОДБОРОДОК, АСИММЕТРИЯ РУК и СКРЕЩИВАНИЕ НОГ.\n"
            "Проверь строго по визуальным данным:\n"
            "1. chin_above_bar: Пересек ли подбородок (нижняя челюсть) уровень грифа перекладины на кадре 3 (PEAK)? "
            "(Если лицо только уперлось в гриф спереди или подбородок ниже перекладины — false).\n"
            "2. knees_straight: Прямые ли ноги в коленях на всех 3 кадрах? (Если голени захлестнуты назад или колени подтянуты — false).\n"
            "3. legs_uncrossed: Не перекрещены ли голени или лодыжки между собой (нет ли наложения стоп внахлест)?\n"
            "4. no_kipping: Выполнено ли подтягивание ровно, без маха ногами, рывка тазом и сильного раскачивания корпуса?\n"
            "5. arms_symmetric: Сгибаются ли руки равномерно без перекоса плеч/локтей и поочередного вытягивания одной рукой?\n\n"
            "Верни ТОЛЬКО валидный JSON по следующей схеме:\n"
            '{"chin_above_bar": bool, "knees_straight": bool, "legs_uncrossed": bool, '
            '"no_kipping": bool, "arms_symmetric": bool, "short_reason": "краткий вывод до 8 слов"}'
        )

        try:
            raw_text = ""
            if GEMINI_SDK == "google-genai":
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=[
                        genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=genai_types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw_text = response.text or ""
            elif GEMINI_SDK == "legacy":
                response = self.client.generate_content(
                    [{"mime_type": "image/jpeg", "data": jpeg_bytes}, prompt],
                    generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
                )
                raw_text = response.text or ""

            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").replace("json", "", 1).strip()

            verdict = json.loads(cleaned)
            updated_errors = set(fsm_errors)

            if not verdict.get("chin_above_bar", True):
                updated_errors.add("подбородок ниже уровня грифа перекладины")
            elif "подбородок ниже уровня грифа перекладины" in updated_errors and min_elbow_deg <= 55.0:
                updated_errors.discard("подбородок ниже уровня грифа перекладины")

            if not verdict.get("knees_straight", True):
                updated_errors.add("нарушение ИП: согнутые в коленях ноги")
            elif "нарушение ИП: согнутые в коленях ноги" in updated_errors:
                updated_errors.discard("нарушение ИП: согнутые в коленях ноги")

            if not verdict.get("legs_uncrossed", True):
                updated_errors.add("нарушение ИП: перекрещенные ноги")
            elif "нарушение ИП: перекрещенные ноги" in updated_errors:
                updated_errors.discard("нарушение ИП: перекрещенные ноги")

            if not verdict.get("no_kipping", True):
                updated_errors.add("подтягивание рывками или с махом ногами/туловищем")
            elif "подтягивание рывками или с махом ногами/туловищем" in updated_errors:
                updated_errors.discard("подтягивание рывками или с махом ногами/туловищем")

            if not verdict.get("arms_symmetric", True):
                updated_errors.add("поочередное (неравномерное) сгибание рук")
            elif "поочередное (неравномерное) сгибание рук" in updated_errors:
                updated_errors.discard("поочередное (неравномерное) сгибание рук")

            note = verdict.get("short_reason", "проверено ИИ-арбитром")
            return updated_errors, note
        except Exception:
            return set(fsm_errors), None


def get_yolo_model():
    """
    Безопасная загрузка модели YOLOv8n-Pose:
    использует ONNX только если установлен модуль onnxruntime,
    иначе автоматически работает на стандартном .pt без ошибок.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        has_onnxruntime = False
        try:
            import onnxruntime  # noqa: F401
            has_onnxruntime = True
        except ImportError:
            has_onnxruntime = False

        base_stem = os.path.splitext(YOLO_MODEL_NAME)[0]
        onnx_file = f"{base_stem}.onnx"

        if has_onnxruntime:
            if not os.path.exists(onnx_file):
                try:
                    pt_model = YOLO(YOLO_MODEL_NAME)
                    pt_model.export(format="onnx", imgsz=CONFIG["yolo_imgsz"], simplify=True)
                except Exception:
                    pass

            if os.path.exists(onnx_file):
                _MODEL_INSTANCE = YOLO(onnx_file, task="pose")
            else:
                _MODEL_INSTANCE = YOLO(YOLO_MODEL_NAME)
        else:
            # Если onnxruntime нет на сервере — сразу загружаем .pt без ошибок
            _MODEL_INSTANCE = YOLO(YOLO_MODEL_NAME)

    return _MODEL_INSTANCE


def select_target_athlete(yolo_result, prev_center=None):
    """Выбирает целевого спортсмена на турнике из всех обнаруженных людей в кадре."""
    if (
        yolo_result is None
        or yolo_result.keypoints is None
        or yolo_result.keypoints.xyn is None
    ):
        return None, prev_center

    xyn_all = yolo_result.keypoints.xyn.cpu().numpy()
    if len(xyn_all) == 0:
        return None, prev_center

    conf_all = (
        yolo_result.keypoints.conf.cpu().numpy()
        if yolo_result.keypoints.conf is not None
        else np.ones((len(xyn_all), 17), dtype=np.float32)
    )
    boxes_xywhn = (
        yolo_result.boxes.xywhn.cpu().numpy()
        if (yolo_result.boxes is not None and yolo_result.boxes.xywhn is not None)
        else None
    )

    best_idx = -1
    best_score = -1e9

    for i in range(len(xyn_all)):
        kpts = xyn_all[i]
        confs = conf_all[i]

        if confs[LM.LEFT_SHOULDER] < 0.25 or confs[LM.RIGHT_SHOULDER] < 0.25:
            continue

        cx = float(0.5 * (kpts[LM.LEFT_SHOULDER][0] + kpts[LM.RIGHT_SHOULDER][0]))
        cy = float(0.5 * (kpts[LM.LEFT_SHOULDER][1] + kpts[LM.RIGHT_SHOULDER][1]))

        if boxes_xywhn is not None and i < len(boxes_xywhn):
            _, _, bw, bh = boxes_xywhn[i]
            area = float(bw * bh)
        else:
            area = float(
                abs(kpts[LM.LEFT_ANKLE][1] - kpts[LM.LEFT_SHOULDER][1])
                * abs(kpts[LM.LEFT_SHOULDER][0] - kpts[LM.RIGHT_SHOULDER][0])
            )

        wrists_y = 0.5 * (kpts[LM.LEFT_WRIST][1] + kpts[LM.RIGHT_WRIST][1])
        hands_up_bonus = 0.35 if wrists_y < cy + 0.05 else -0.25
        center_dist = math.hypot(cx - 0.5, cy - 0.45)

        if prev_center is not None:
            track_dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
            if track_dist > 0.28:
                continue
            score = -3.0 * track_dist + 1.5 * area + hands_up_bonus
        else:
            score = 2.2 * area - 0.8 * center_dist + hands_up_bonus

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None, prev_center

    chosen_kpts = xyn_all[best_idx]
    chosen_confs = conf_all[best_idx]
    landmarks = []
    for pt_idx in range(17):
        x, y = float(chosen_kpts[pt_idx][0]), float(chosen_kpts[pt_idx][1])
        c = float(chosen_confs[pt_idx])
        landmarks.append(StabilizedLandmark(x, y, 0.0, visibility=c, presence=c))

    new_center = (
        0.5 * (landmarks[LM.LEFT_SHOULDER].x + landmarks[LM.RIGHT_SHOULDER].x),
        0.5 * (landmarks[LM.LEFT_SHOULDER].y + landmarks[LM.RIGHT_SHOULDER].y),
    )
    return landmarks, new_center


class OneEuroFilter:
    """Адаптивный фильтр низких частот 1€ (One Euro Filter)."""

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
    """Стабилизатор скелета с анатомической защитой от схлопывания и 1€-фильтрацией."""

    HAND_INDICES = {LM.LEFT_WRIST, LM.RIGHT_WRIST}

    def __init__(self, config=CONFIG):
        self.cfg = config
        self.filters_x = {}
        self.filters_y = {}
        self.prev_pts = None

        self.orientation_sign = None
        self.sh_w_history = deque(maxlen=25)
        self.hip_w_history = deque(maxlen=25)
        self.wrist_w_history = deque(maxlen=25)

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
        return self.filters_x[idx], self.filters_y[idx]

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
            StabilizedLandmark(lm.x, lm.y, lm.z, lm.visibility, lm.presence)
            for lm in raw_landmarks
        ]

        if self.prev_pts is not None:
            max_jump = self.cfg["max_joint_jump_per_frame"]
            for idx, p in enumerate(pts):
                prev_p = self.prev_pts[idx]
                conf = p.visibility
                if (p.x <= 1e-4 and p.y <= 1e-4) or conf < 0.20:
                    p.x, p.y = prev_p.x, prev_p.y
                    continue

                dist = math.hypot(p.x - prev_p.x, p.y - prev_p.y)
                if conf < 0.45:
                    p.x = prev_p.x + 0.25 * (p.x - prev_p.x)
                    p.y = prev_p.y + 0.25 * (p.y - prev_p.y)
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
                    pts, LM.LEFT_WRIST, LM.RIGHT_WRIST,
                    min_width=self.ref_wrist_w * 0.78,
                    target_width=self.ref_wrist_w,
                )

            sh_collapsed = self._restore_pair_width(
                pts, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
                min_width=self.ref_sh_w * collapse_ratio,
                target_width=self.ref_sh_w,
            )

            target_hip_w = self.ref_hip_w if self.ref_hip_w is not None else (self.ref_sh_w * 0.62)
            hip_collapsed = self._restore_pair_width(
                pts, LM.LEFT_HIP, LM.RIGHT_HIP,
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
                        pts, LM.LEFT_KNEE, LM.RIGHT_KNEE,
                        min_width=0.65 * prev_kn_w,
                        target_width=prev_kn_w,
                    )

        smoothed_pts = []
        for idx, p in enumerate(pts):
            fx, fy = self._get_filters(idx)
            if hands_on_bar and idx in self.HAND_INDICES:
                sx = fx.filter(p.x, timestamp_sec, min_cutoff_override=0.18, beta_override=0.008)
                sy = fy.filter(p.y, timestamp_sec, min_cutoff_override=0.25, beta_override=0.015)
            else:
                sx = fx.filter(p.x, timestamp_sec)
                sy = fy.filter(p.y, timestamp_sec)
            smoothed_pts.append(StabilizedLandmark(sx, sy, 0.0, p.visibility, p.presence))

        self.prev_pts = smoothed_pts
        return smoothed_pts


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle


def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


def compute_bar_y_from_hands(lm, ref_sh_w=None):
    """
    Вычисляет положение оси перекладины по костяшкам/запястьям с опорой на эталонную ширину плеч.
    """
    l_wr, r_wr = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
    if l_wr.visibility >= 0.35 and r_wr.visibility >= 0.35:
        wrist_y = 0.5 * (l_wr.y + r_wr.y)
    elif l_wr.visibility > r_wr.visibility:
        wrist_y = l_wr.y
    else:
        wrist_y = r_wr.y

    if ref_sh_w is not None and ref_sh_w > 0.06:
        sh_w = ref_sh_w
    else:
        sh_w = max(0.08, abs(lm[LM.LEFT_SHOULDER].x - lm[LM.RIGHT_SHOULDER].x))
    return float(max(0.02, wrist_y - 0.18 * sh_w))


def compute_anatomical_chin(lm, base_shoulder_w=0.12):
    nose_x, nose_y = lm[LM.NOSE].x, lm[LM.NOSE].y
    shoulder_y = 0.5 * (lm[LM.LEFT_SHOULDER].y + lm[LM.RIGHT_SHOULDER].y)
    shoulder_x = 0.5 * (lm[LM.LEFT_SHOULDER].x + lm[LM.RIGHT_SHOULDER].x)

    l_eye, r_eye = lm[LM.LEFT_EYE], lm[LM.RIGHT_EYE]
    if l_eye.visibility > 0.3 and r_eye.visibility > 0.3 and l_eye.y > 0 and r_eye.y > 0:
        eye_y = 0.5 * (l_eye.y + r_eye.y)
        eye_to_nose = max(0.01, nose_y - eye_y)
        chin_drop = np.clip(1.35 * eye_to_nose, 0.20 * base_shoulder_w, 0.34 * base_shoulder_w)
    else:
        chin_drop = 0.25 * base_shoulder_w

    chin_y = max(nose_y + chin_drop, shoulder_y - 0.34 * base_shoulder_w)
    chin_x = 0.80 * nose_x + 0.20 * shoulder_x
    return float(chin_x), float(chin_y)


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


def draw_hud(
    frame,
    valid_reps,
    total_attempts,
    state,
    hang_sec,
    elbow_deg,
    knee_deg,
    bar_y,
    chin_above_now,
    active_errors,
    current_rep_idx=1,
    ai_calls_count=0,
    ai_dispute_live=False,
    ai_banner_active=False,
    ai_last_rep_num=None,
    ai_last_valid=None,
):
    """Отрисовка HUD-панели, динамической линии перекладины для каждого повторения и индикатора ИИ-арбитра."""
    h, w, _ = frame.shape
    if bar_y is not None:
        bar_px = int(bar_y * h)
        bar_color = (0, 255, 0) if chin_above_now else (0, 215, 255)
        cv2.line(frame, (0, bar_px), (w, bar_px), bar_color, 3)
        label_y = bar_px + 22 if bar_px < 35 else bar_px - 8
        rep_tag = f"REP #{current_rep_idx}" if state != "WAIT_HANG" else f"NEXT REP #{current_rep_idx}"
        label_text = f"BAR LINE [{rep_tag}: Y={bar_y:.3f}] {'[CHIN ABOVE]' if chin_above_now else ''}"
        cv2.putText(frame, label_text, (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
        cv2.putText(frame, label_text, (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, bar_color, 2)

    # 1. ОТДЕЛЬНЫЙ ВЕРХНИЙ ИНДИКАТОР-БАННЕР ПРИ СРАБАТЫВАНИИ ИИ-АРБИТРА
    if ai_banner_active and ai_last_rep_num is not None:
        banner_overlay = frame.copy()
        bw, bh = min(w - 20, 390), 58
        bx1 = max(10, w - bw - 12)
        by1 = 12
        bg_color = (25, 95, 25) if ai_last_valid else (45, 20, 110)
        border_color = (0, 255, 120) if ai_last_valid else (255, 80, 255)

        cv2.rectangle(banner_overlay, (bx1, by1), (bx1 + bw, by1 + bh), bg_color, -1)
        cv2.addWeighted(banner_overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx1 + bw, by1 + bh), border_color, 2)

        verdict_tag = "VALID (OK)" if ai_last_valid else "REJECTED (ERR)"
        cv2.putText(
            frame,
            f"AI REFEREE TRIGGERED [REP #{ai_last_rep_num}]",
            (bx1 + 12, by1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"VERDICT: {verdict_tag} | TOTAL CALLS: {ai_calls_count}",
            (bx1 + 12, by1 + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 2. ОСНОВНАЯ ПАНЕЛЬ HUD ВНИЗУ С ПОСТОЯННЫМ ИНДИКАТОРОМ ИИ-АРБИТРА
    overlay = frame.copy()
    box_w, box_h = min(w - 20, 460), 158
    y_top = max(10, h - box_h - 15)
    cv2.rectangle(overlay, (10, y_top), (10 + box_w, y_top + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    cv2.putText(frame, f"VALID REPS: {valid_reps} / ATTEMPTS: {total_attempts}", (20, y_top + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 120), 2)
    cv2.putText(frame, f"STATE: {state} | HANG: {hang_sec:.2f}s", (20, y_top + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"ELBOW: {int(elbow_deg)} deg | KNEE: {int(knee_deg)} deg", (20, y_top + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 2)

    err_status = "ERRORS IN REP: YES" if active_errors else "TECHNIQUE: OK"
    err_color = (0, 80, 255) if active_errors else (0, 255, 0)
    cv2.putText(frame, err_status, (20, y_top + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, err_color, 2)

    # Строка 5: Индикатор состояния ИИ-арбитра
    if ai_banner_active and ai_last_rep_num is not None:
        ai_text = f"AI REFEREE: TRIGGERED ON REP #{ai_last_rep_num} (CALLS: {ai_calls_count})"
        ai_color = (255, 120, 255)
    elif ai_dispute_live:
        ai_text = f"AI REFEREE: DISPUTE ARMED (CALLS: {ai_calls_count})"
        ai_color = (0, 215, 255)
    else:
        ai_text = f"AI REFEREE: STANDBY | CALLS: {ai_calls_count}"
        ai_color = (180, 180, 180) if ai_calls_count == 0 else (255, 200, 100)

    cv2.putText(frame, ai_text, (20, y_top + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ai_color, 2, cv2.LINE_AA)


def get_calibration_preview(video_path, manual_bar_y=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, 0.25

    h, w, _ = frame.shape
    detected_bar_y = 0.25

    model = get_yolo_model()
    preds = model.predict(frame, conf=0.35, imgsz=CONFIG["yolo_imgsz"], verbose=False)
    if preds:
        lm, _ = select_target_athlete(preds[0])
        if lm is not None:
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
    force_ai_all_reps=False,
    progress_callback=None
):
    result = {
        "total_attempts": 0,
        "valid_reps": 0,
        "bar_y": round(manual_bar_y if manual_bar_y is not None else 0.12, 3),
        "errors": [],
        "reps_detail": [],
        "comment": "Повторения не зафиксированы.",
        "tracked_video_path": None,
        "ai_referee_calls": 0,
        "ai_referee_triggered": False,
        "ai_verified_reps": []
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["comment"] = "Ошибка открытия видеофайла."
        return result

    writer = None
    raw_out_path = None
    reps_log = []
    bar_y = manual_bar_y if manual_bar_y is not None else 0.12

    ai_calls_count = 0
    ai_last_trigger_time = -999.0
    ai_last_rep_num = None
    ai_last_valid = None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ПУНКТ 1: Нормализация частоты анализа до 15 к/с и пересчет порогов под эффективный FPS
        frame_step = max(1, int(round(fps / CONFIG["target_eval_fps"])))
        eff_fps = fps / frame_step
        fps_scale = eff_fps / 30.0

        min_asym_f = max(2, int(round(CONFIG["asym_min_frames"] * fps_scale)))
        min_knee_f = max(2, int(round(CONFIG["knee_err_min_frames"] * fps_scale)))
        min_cross_f = max(2, int(round(CONFIG["cross_err_min_frames"] * fps_scale)))
        min_feet_f = max(3, int(round(CONFIG["feet_apart_min_frames"] * fps_scale)))
        min_kip_f = max(3, int(round(CONFIG["kipping_min_frames"] * fps_scale)))

        scale = min(1.0, 720.0 / max(orig_w, orig_h, 1))
        out_w = max(2, int(orig_w * scale) // 2 * 2)
        out_h = max(2, int(orig_h * scale) // 2 * 2)

        if generate_tracked_video:
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            raw_out_path = tmp_out.name
            tmp_out.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Записываем выходное видео с эффективной частотой eff_fps (в реальном времени без дублей)
            writer = cv2.VideoWriter(raw_out_path, fourcc, eff_fps, (out_w, out_h))

        state = "WAIT_HANG"
        hang_start_time = None
        last_hang_duration = 0.0
        pre_bend_frames = 0

        # Буферы для индивидуального обновления линии перекладины на КАЖДОЕ подтягивание
        hang_bar_samples = deque(maxlen=CONFIG["bar_hang_window_frames"])
        rep_bar_samples = deque(maxlen=CONFIG["bar_rep_window_frames"])
        rep_bar_anchor_y = bar_y

        shoulder_w_samples = []
        base_sh_tilt_samples = []
        base_el_tilt_samples = []
        base_el_diff_samples = []

        base_shoulder_w = 0.12
        base_sh_tilt = 0.03
        base_el_tilt = 0.03
        base_el_diff = 5.0

        stabilizer = SkeletonStabilizer(CONFIG)
        verifier = GeminiDisputeVerifier(enabled=CONFIG["enable_gemini_verifier"])

        chin_x_filter = OneEuroFilter(
            min_cutoff=CONFIG["filter_min_cutoff_x"],
            beta=CONFIG["filter_beta_x"]
        )
        chin_y_filter = OneEuroFilter(
            min_cutoff=CONFIG["filter_min_cutoff_y"],
            beta=CONFIG["filter_beta_y"]
        )

        # На 15 FPS окно медианного сглаживания в 2 кадра дает мгновенный отклик без задержки в пике
        buf_len = 2 if eff_fps <= 18.0 else 3
        chin_buf = deque(maxlen=buf_len)
        nose_buf = deque(maxlen=buf_len)
        elbow_buf = deque(maxlen=buf_len)
        knee_buf = deque(maxlen=buf_len)
        shoulder_y_buf = deque(maxlen=buf_len)

        pre_pull_window = max(8, int(round(0.8 * eff_fps)))
        pre_pull_knee_err = deque(maxlen=pre_pull_window)
        pre_pull_cross_err = deque(maxlen=pre_pull_window)

        rep_start_time = 0.0
        min_elbow_in_rep = 180.0
        min_shoulder_y_in_rep = 1.0
        min_chin_y_in_rep = 1.0
        chin_above_frames = 0
        rep_errors = set()

        last_clean_hang_frame = None
        rep_start_frame = None
        rep_mid_frame = None
        rep_peak_frame = None

        rep_frames = 0
        asym_frames = 0
        knee_bent_frames = 0
        legs_crossed_frames = 0
        feet_apart_frames = 0
        kipping_frames = 0
        rep_ankle_x = []

        frame_idx = 0
        eval_idx = 0
        tracked_athlete_center = None
        model = get_yolo_model()
        yolo_imgsz = CONFIG["yolo_imgsz"]

        while cap.isOpened():
            # Быстрый пропуск промежуточных кадров без декодирования матрицы изображения
            if (frame_idx + 1) % frame_step != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            eval_idx += 1
            current_time = frame_idx / fps

            if progress_callback and total_frames > 0 and eval_idx % 5 == 0:
                progress_callback(min(1.0, frame_idx / total_frames))

            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            clean_frame = frame.copy()

            # ПУНКТ 2: Быстрый инференс YOLOv8n-Pose с фиксированным imgsz=480
            preds = model.predict(frame, conf=0.35, imgsz=yolo_imgsz, verbose=False)
            raw_lm, tracked_athlete_center = (
                select_target_athlete(preds[0], tracked_athlete_center)
                if preds
                else (None, tracked_athlete_center)
            )

            mean_elbow_angle = 0.0
            mean_knee_angle = 180.0
            chin_above_now = False

            if raw_lm is not None:
                lm = stabilizer.process(raw_lm, current_time, bar_y=bar_y)

                def pt(idx):
                    return [lm[idx].x, lm[idx].y]

                l_sh, r_sh = pt(LM.LEFT_SHOULDER), pt(LM.RIGHT_SHOULDER)
                l_el, r_el = pt(LM.LEFT_ELBOW), pt(LM.RIGHT_ELBOW)
                l_wr, r_wr = pt(LM.LEFT_WRIST), pt(LM.RIGHT_WRIST)
                l_hip, r_hip = pt(LM.LEFT_HIP), pt(LM.RIGHT_HIP)
                l_kn, r_kn = pt(LM.LEFT_KNEE), pt(LM.RIGHT_KNEE)
                l_an, r_an = pt(LM.LEFT_ANKLE), pt(LM.RIGHT_ANKLE)

                wrist_y = (l_wr[1] + r_wr[1]) / 2.0
                wrists_conf_ok = (lm[LM.LEFT_WRIST].visibility >= 0.30 or lm[LM.RIGHT_WRIST].visibility >= 0.30)

                instant_bar_y = compute_bar_y_from_hands(lm, ref_sh_w=base_shoulder_w)

                if manual_bar_y is None and not hang_bar_samples and not rep_bar_samples:
                    hang_bar_samples.append(instant_bar_y)
                    bar_y = float(np.median(hang_bar_samples))
                    rep_bar_anchor_y = bar_y

                nose = pt(LM.NOSE)
                raw_chin_x, raw_chin_y = compute_anatomical_chin(lm, base_shoulder_w)

                chin_x = chin_x_filter.filter(raw_chin_x, current_time)
                filtered_chin_y = chin_y_filter.filter(raw_chin_y, current_time)
                raw_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
                shoulder_center_x = (l_sh[0] + r_sh[0]) / 2.0

                angle_el_l = calculate_angle(l_sh, l_el, l_wr)
                angle_el_r = calculate_angle(r_sh, r_el, r_wr)
                raw_elbow = (angle_el_l + angle_el_r) / 2.0

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
                    nose_y + 0.18 * base_shoulder_w,
                    shoulder_y - 0.34 * base_shoulder_w
                )
                mean_elbow_angle = float(np.median(elbow_buf))
                mean_knee_angle = float(np.median(knee_buf))

                mean_hip_angle = (calculate_angle(l_sh, l_hip, l_kn) + calculate_angle(r_sh, r_hip, r_kn)) / 2.0
                curr_sh_w = abs(l_sh[0] - r_sh[0])

                if mean_elbow_angle >= 170.0 and curr_sh_w > 0.05 and len(shoulder_w_samples) < 30:
                    shoulder_w_samples.append(curr_sh_w)
                    base_shoulder_w = float(np.median(shoulder_w_samples))
                    base_sh_tilt_samples.append(abs(l_sh[1] - r_sh[1]) / max(0.06, base_shoulder_w))
                    base_el_tilt_samples.append(abs(l_el[1] - r_el[1]) / max(0.06, base_shoulder_w))
                    base_el_diff_samples.append(abs(angle_el_l - angle_el_r))

                    base_sh_tilt = float(np.median(base_sh_tilt_samples))
                    base_el_tilt = float(np.median(base_el_tilt_samples))
                    base_el_diff = float(np.median(base_el_diff_samples))

                curr_sh_tilt = abs(l_sh[1] - r_sh[1]) / max(0.06, base_shoulder_w)
                curr_el_tilt = abs(l_el[1] - r_el[1]) / max(0.06, base_shoulder_w)
                elbow_diff = abs(angle_el_l - angle_el_r)

                net_sh_tilt = max(0.0, curr_sh_tilt - base_sh_tilt)
                net_el_tilt = max(0.0, curr_el_tilt - base_el_tilt)
                net_el_diff = max(0.0, elbow_diff - base_el_diff)

                head_fully_in_frame = nose_y > 0.04
                is_asym_frame = (
                    head_fully_in_frame
                    and (15.0 < mean_elbow_angle < 164.0)
                    and (
                        net_el_tilt > CONFIG["asym_elbow_tilt_ratio"]
                        or net_sh_tilt > CONFIG["asym_shoulder_tilt_ratio"]
                        or (net_el_diff > CONFIG["asym_elbow_diff_deg"] and net_el_tilt > 0.08)
                    )
                )

                ankle_dx = l_an[0] - r_an[0]
                knee_dx = l_kn[0] - r_kn[0]
                hip_dx = l_hip[0] - r_hip[0]
                hip_w = max(0.04, abs(hip_dx))
                ankle_dist = abs(ankle_dx)
                knee_dist = abs(knee_dx)
                ankle_dy = abs(l_an[1] - r_an[1])

                shins_crossed_x = (
                    (
                        ankle_dx * hip_dx < CONFIG["cross_product_threshold"]
                        or ankle_dx * knee_dx < CONFIG["cross_product_threshold"]
                    )
                    and (ankle_dist > hip_w * CONFIG["cross_min_overlap_ratio"])
                )
                ankles_stacked_vertically = (
                    knee_dist > hip_w * 0.45
                    and (ankle_dist < knee_dist * 0.45 or ankle_dist < hip_w * 0.28)
                    and ankle_dy > hip_w * 0.20
                )
                legs_crossed = shins_crossed_x or ankles_stacked_vertically

                feet_apart = ankle_dist > (base_shoulder_w * CONFIG["feet_apart_ratio"])
                knees_bent = (
                    (mean_knee_angle < CONFIG["knee_bent_threshold"])
                    or (shin_ratio < CONFIG["shin_ratio_threshold"])
                )

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

                hands_on_bar = abs(wrist_y - bar_y) < CONFIG["bar_release_dist"]

                # --- ОБНОВЛЕНИЕ ЛИНИИ ПЕРЕКЛАДИНЫ НА КАЖДОЕ ПОДТЯГИВАНИЕ ---
                if manual_bar_y is None and hands_on_bar and wrists_conf_ok:
                    if state == "WAIT_HANG":
                        hang_bar_samples.append(instant_bar_y)
                        bar_y = float(np.median(hang_bar_samples))
                        rep_bar_anchor_y = bar_y
                    elif state in ("PULLING", "DESCENDING"):
                        if abs(instant_bar_y - rep_bar_anchor_y) <= CONFIG["bar_max_Rep_drift"]:
                            rep_bar_samples.append(instant_bar_y)
                        if rep_bar_samples:
                            bar_y = float(0.65 * rep_bar_anchor_y + 0.35 * float(np.median(rep_bar_samples)))

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

                if eval_idx <= 3 and state == "WAIT_HANG" and mean_elbow_angle < 65.0 and hands_on_bar:
                    state = "DESCENDING"
                    rep_start_time = 0.0
                    if manual_bar_y is None:
                        rep_bar_samples.clear()
                        rep_bar_samples.append(instant_bar_y)
                        bar_y = instant_bar_y
                        rep_bar_anchor_y = bar_y
                    min_elbow_in_rep = mean_elbow_angle
                    min_shoulder_y_in_rep = shoulder_y
                    min_chin_y_in_rep = chin_y
                    chin_above_frames = CONFIG["min_chin_above_frames"] if chin_above_now else 0
                    rep_errors = set()
                    rep_frames = 1
                    asym_frames = knee_bent_frames = legs_crossed_frames = feet_apart_frames = kipping_frames = 0
                    rep_ankle_x = [ankle_center_x]
                    rep_start_frame = clean_frame
                    rep_mid_frame = clean_frame
                    rep_peak_frame = clean_frame

                elif state == "WAIT_HANG":
                    if hands_on_bar:
                        pre_pull_knee_err.append(1 if knees_bent else 0)
                        pre_pull_cross_err.append(1 if legs_crossed else 0)

                    if mean_elbow_angle >= elbow_ext_angle and hands_on_bar:
                        if hang_start_time is None:
                            hang_start_time = current_time
                        last_hang_duration = current_time - hang_start_time
                        pre_bend_frames = 0
                        last_clean_hang_frame = clean_frame
                    else:
                        if (
                            CONFIG["pull_start_angle"] <= mean_elbow_angle < elbow_ext_angle
                            and hands_on_bar
                        ):
                            pre_bend_frames += 1

                        if mean_elbow_angle < CONFIG["pull_start_angle"] and hands_on_bar:
                            state = "PULLING"
                            rep_start_time = current_time

                            if manual_bar_y is None:
                                if hang_bar_samples:
                                    rep_bar_anchor_y = float(np.median(hang_bar_samples))
                                else:
                                    rep_bar_anchor_y = instant_bar_y
                                rep_bar_samples.clear()
                                rep_bar_samples.append(rep_bar_anchor_y)
                                bar_y = rep_bar_anchor_y

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

                            rep_start_frame = last_clean_hang_frame if last_clean_hang_frame is not None else clean_frame
                            rep_mid_frame = clean_frame
                            rep_peak_frame = clean_frame

                            if pre_bend_frames > int(CONFIG["pre_bend_max_time"] * eff_fps):
                                rep_errors.add("нарушение ИП: согнутые в локтях руки перед стартом")

                            if sum(pre_pull_knee_err) >= min_knee_f:
                                rep_errors.add("нарушение ИП: согнутые в коленях ноги")
                            if sum(pre_pull_cross_err) >= min_cross_f:
                                rep_errors.add("нарушение ИП: перекрещенные ноги")

                            is_first_rep_early = (len(reps_log) == 0 and rep_start_time < 2.0)
                            if not is_first_rep_early and last_hang_duration < min_hang_sec:
                                rep_errors.add(f"фиксация ИП менее 1.0 сек ({last_hang_duration:.2f} с)")

                            hang_start_time = None
                            pre_bend_frames = 0

                elif state in ("PULLING", "DESCENDING"):
                    if not hands_on_bar:
                        rep_duration = current_time - rep_start_time
                        if rep_duration >= 0.40 and min_elbow_in_rep <= 115.0:
                            if (
                                chin_above_frames < CONFIG["min_chin_above_frames"]
                                or min_elbow_in_rep > CONFIG["max_peak_elbow_angle"]
                            ):
                                rep_errors.add("подбородок ниже уровня грифа перекладины")
                            rep_errors.add("срыв / спрыгивание с перекладины без фиксации виса")
                            reps_log.append({
                                "rep_number": len(reps_log) + 1,
                                "timestamp": f"{format_timestamp(rep_start_time)} – {format_timestamp(current_time)}",
                                "bar_y": round(bar_y, 3),
                                "valid": False,
                                "status_text": "❌ Не засчитано",
                                "ai_triggered": False,
                                "ai_indicator": "⚪ Не требовался",
                                "ai_verdict": "—",
                                "errors": sorted(rep_errors)
                            })
                        state = "WAIT_HANG"
                        hang_start_time = None
                        last_hang_duration = 0.0
                        pre_bend_frames = 0
                        rep_errors = set()
                        pre_pull_knee_err.clear()
                        pre_pull_cross_err.clear()
                        hang_bar_samples.clear()
                        rep_bar_samples.clear()
                    else:
                        rep_frames += 1

                        if state == "PULLING" and 75.0 <= mean_elbow_angle <= 110.0:
                            rep_mid_frame = clean_frame
                        if chin_y <= min_chin_y_in_rep or mean_elbow_angle <= min_elbow_in_rep:
                            rep_peak_frame = clean_frame

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

                            if (
                                rep_duration >= CONFIG["min_rep_duration"]
                                and min_elbow_in_rep <= CONFIG["min_pull_depth_angle"]
                            ):
                                ankle_swing_span = (max(rep_ankle_x) - min(rep_ankle_x)) if rep_ankle_x else 0.0

                                asym_thresh = max(min_asym_f, int(rep_frames * CONFIG["asym_frame_ratio"]))
                                knee_thresh = max(min_knee_f, int(rep_frames * CONFIG["knee_err_frame_ratio"]))
                                cross_thresh = min_cross_f
                                feet_thresh = max(min_feet_f, int(rep_frames * CONFIG["feet_apart_frame_ratio"]))
                                kip_thresh = max(min_kip_f, int(rep_frames * CONFIG["kipping_frame_ratio"]))

                                if asym_frames >= asym_thresh:
                                    rep_errors.add("поочередное (неравномерное) сгибание рук")
                                if knee_bent_frames >= knee_thresh:
                                    rep_errors.add("нарушение ИП: согнутые в коленях ноги")
                                if legs_crossed_frames >= cross_thresh:
                                    rep_errors.add("нарушение ИП: перекрещенные ноги")
                                if feet_apart_frames >= feet_thresh:
                                    rep_errors.add("нарушение ИП: ступни не вместе (ноги разведены)")
                                if (
                                    kipping_frames >= kip_thresh
                                    or ankle_swing_span > (base_shoulder_w * CONFIG["ankle_swing_span_ratio"])
                                ):
                                    rep_errors.add("подтягивание рывками или с махом ногами/туловищем")
                                if (
                                    chin_above_frames < CONFIG["min_chin_above_frames"]
                                    or min_elbow_in_rep > CONFIG["max_peak_elbow_angle"]
                                ):
                                    rep_errors.add("подбородок ниже уровня грифа перекладины")

                                # --- ВЕРИФИКАЦИЯ СПОРНЫХ ПОПЫТОК ИЛИ ПОЛНЫЙ АУДИТ ЧЕРЕЗ ИИ-АРБИТРА ---
                                ai_referee_note = None
                                dispute_reasons = verifier.get_dispute_reasons(
                                    rep_errors=rep_errors,
                                    min_chin_y=min_chin_y_in_rep,
                                    bar_y=bar_y,
                                    min_elbow_deg=min_elbow_in_rep,
                                    asym_frames=asym_frames,
                                    asym_thresh=asym_thresh,
                                    knee_frames=knee_bent_frames,
                                    knee_thresh=knee_thresh,
                                    cross_frames=legs_crossed_frames,
                                    cross_thresh=cross_thresh,
                                    kip_frames=kipping_frames,
                                    kip_thresh=kip_thresh,
                                )

                                should_run_ai = verifier.enabled and (len(dispute_reasons) > 0 or force_ai_all_reps)

                                if should_run_ai:
                                    if force_ai_all_reps and not dispute_reasons:
                                        dispute_reasons = ["ПОЛНЫЙ АУДИТ"]

                                    rep_errors, ai_referee_note = verifier.verify_repetition(
                                        start_frame=rep_start_frame,
                                        mid_frame=rep_mid_frame,
                                        peak_frame=rep_peak_frame,
                                        bar_y=bar_y,
                                        fsm_errors=rep_errors,
                                        min_elbow_deg=min_elbow_in_rep,
                                    )

                                is_valid = len(rep_errors) == 0
                                current_rep_num = len(reps_log) + 1
                                ai_was_triggered = ai_referee_note is not None

                                if ai_was_triggered:
                                    ai_calls_count += 1
                                    ai_last_trigger_time = current_time
                                    ai_last_rep_num = current_rep_num
                                    ai_last_valid = is_valid

                                status_str = "✅ Засчитано" if is_valid else "❌ Не засчитано"
                                if ai_was_triggered:
                                    status_str += " (🤖 ИИ-арбитр)"

                                reps_log.append({
                                    "rep_number": current_rep_num,
                                    "timestamp": f"{format_timestamp(rep_start_time)} – {format_timestamp(current_time)}",
                                    "bar_y": round(bar_y, 3),
                                    "valid": is_valid,
                                    "status_text": status_str,
                                    "ai_triggered": ai_was_triggered,
                                    "ai_indicator": f"🤖 Сработал ({', '.join(dispute_reasons)})" if ai_was_triggered else "⚪ Локально (YOLO)",
                                    "ai_verdict": ai_referee_note if ai_was_triggered else "—",
                                    "errors": sorted(rep_errors) if rep_errors else ["Нет ошибок"]
                                })

                            state = "WAIT_HANG"
                            hang_start_time = current_time
                            last_hang_duration = 0.0
                            pre_bend_frames = 0
                            rep_errors = set()
                            pre_pull_knee_err.clear()
                            pre_pull_cross_err.clear()
                            if manual_bar_y is None:
                                hang_bar_samples.clear()
                                rep_bar_samples.clear()
                                if hands_on_bar and wrists_conf_ok:
                                    hang_bar_samples.append(instant_bar_y)
                                    bar_y = instant_bar_y
                                    rep_bar_anchor_y = bar_y

                if writer is not None:
                    draw_skeleton(frame, lm, chin_pt=(chin_x, chin_y))

            if writer is not None:
                valid_so_far = sum(1 for r in reps_log if r["valid"])
                has_active_err = (
                    state != "WAIT_HANG"
                    and (
                        bool(rep_errors)
                        or knee_bent_frames >= min_knee_f
                        or legs_crossed_frames >= min_cross_f
                        or asym_frames >= min_asym_f
                        or kipping_frames >= min_kip_f
                    )
                )
                ai_dispute_live = (
                    state != "WAIT_HANG"
                    and (
                        asym_frames >= 1
                        or legs_crossed_frames >= 1
                        or knee_bent_frames >= 1
                        or abs(min_chin_y_in_rep - bar_y) <= CONFIG["dispute_chin_margin"]
                    )
                )
                ai_banner_active = (current_time - ai_last_trigger_time) <= CONFIG["ai_banner_display_sec"]

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
                    active_errors=has_active_err,
                    current_rep_idx=len(reps_log) + 1,
                    ai_calls_count=ai_calls_count,
                    ai_dispute_live=ai_dispute_live,
                    ai_banner_active=ai_banner_active,
                    ai_last_rep_num=ai_last_rep_num,
                    ai_last_valid=ai_last_valid,
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
    ai_verified_list = [r["rep_number"] for r in reps_log if r.get("ai_triggered")]

    summary_lines = [
        f"Повторение #{r['rep_number']} [{r['timestamp']}] (Гриф Y={r.get('bar_y', bar_y):.3f}): "
        f"{r['status_text']} | ИИ-арбитр: {r.get('ai_indicator', '—')} "
        f"({', '.join(r['errors'])})"
        for r in reps_log
    ]

    result["total_attempts"] = len(reps_log)
    result["valid_reps"] = valid_count
    result["bar_y"] = round(bar_y if bar_y is not None else 0.12, 3)
    result["errors"] = all_unique_errors
    result["reps_detail"] = reps_log
    result["comment"] = "\n".join(summary_lines) if summary_lines else "Повторения не зафиксированы."
    result["tracked_video_path"] = final_video_path
    result["ai_referee_calls"] = ai_calls_count
    result["ai_referee_triggered"] = ai_calls_count > 0
    result["ai_verified_reps"] = ai_verified_list
    return result
