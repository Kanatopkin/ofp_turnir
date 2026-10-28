from collections import deque
import base64
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.error
import cv2
import numpy as np
from ultralytics import YOLO

# Безопасный импорт SDK для Gemini API
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

YOLO_MODEL_NAME = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
GEMINI_MODEL_NAME = os.getenv("GEMINI_REFEREE_MODEL", "gemini-1.5-flash")
GEMINI_FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]
_MODEL_INSTANCE = None

CONFIG = {
    "target_eval_fps": 15.0,
    "yolo_imgsz": 480,
    # 1. Параметры виса и старта (регламент ГТО / ВсОШ)
    "min_hang_sec": 0.72,
    "elbow_ext_angle": 153.0,
    "pre_bend_max_time": 0.75,
    "pull_start_angle": 144.0,
    "min_pull_depth_angle": 120.0,
    "min_rep_duration": 0.45,
    "min_shoulder_rise_for_rep": 0.022,
    # 2. Фиксация положения подбородка относительно грифа
    "chin_bar_tolerance": 0.024,
    "narrow_grip_max_elbow": 66.0,
    "wide_grip_max_elbow": 98.0,
    "nose_bar_margin": 0.005,
    "narrow_grip_shoulder_drop": 0.052,
    "wide_grip_shoulder_drop": 0.092,
    "min_chin_above_frames": 1,
    # 3. Контроль техники ног, корпуса и головы
    "knee_bent_threshold": 160.0,
    "shin_ratio_threshold": 0.70,
    "thigh_foreshorten_ratio": 0.46,
    "knee_err_frame_ratio": 0.18,
    "knee_err_min_frames": 2,
    "cross_product_threshold": -0.0006,
    "cross_min_overlap_ratio": 0.22,
    "cross_err_min_frames": 3,
    "feet_apart_ratio": 1.20,
    "feet_apart_frame_ratio": 0.25,
    "feet_apart_min_frames": 3,
    "hip_kipping_angle": 150.0,
    "center_body_bend_angle": 162.0,
    "hip_swing_ratio": 0.50,            # Чувствительность к раскачиванию таза
    "ankle_swing_ratio": 0.42,          # Фиксирует мах ногами
    "ankle_swing_span_ratio": 0.45,     # Фиксирует заброс ног в сторону
    "head_thrown_back_ear_drop": 0.042, # Отсекает нормальный взгляд вверх, ловит только сильное запрокидывание лица в потолок
    "head_thrown_back_min_frames": 4,   # Требует устойчивого удержания запрокинутой головы (от 0.27 с)
    "kipping_frame_ratio": 0.15,
    "kipping_min_frames": 2,
    "asym_elbow_diff_deg": 24.0,        # Разница углов локтей
    "asym_combo_elbow_deg": 18.0,       # Разница углов при наклоне плеч
    "asym_elbow_tilt_ratio": 0.12,      # Вертикальный перекос локтей
    "asym_shoulder_tilt_ratio": 0.115,  # Вертикальный перекос плеч
    "asym_frame_ratio": 0.16,
    "asym_min_frames": 2,
    # 4. Стабилизация перекладины и фильтрация шума
    "bar_release_dist": 0.20,
    "release_confirm_frames": 6,
    "bar_calibration_frames": 10,
    "wrist_lock_max_drift_y": 0.035,
    "wrist_lock_max_drift_x": 0.045,
    "filter_min_cutoff_x": 0.55,
    "filter_beta_x": 0.025,
    "filter_min_cutoff_y": 0.95,
    "filter_beta_y": 0.15,
    "skeleton_collapse_ratio": 0.72,
    "max_joint_jump_per_frame": 0.08,
    # 5. Параметры ИИ-арбитра
    "enable_gemini_verifier": True,
    "dispute_chin_margin": 0.022,
    "dispute_frame_margin": 1,
    "ai_banner_display_sec": 3.0,
}


class LM:
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
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(presence)


def _resolve_gemini_api_key():
    def _clean(val):
        if not val:
            return None
        s = str(val).strip().strip("'").strip('"').strip()
        try:
            s.encode("ascii")
            return s if len(s) >= 25 else None
        except UnicodeEncodeError:
            return None

    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        cleaned = _clean(os.getenv(env_name))
        if cleaned:
            return cleaned

    try:
        import streamlit as st
        secrets_dict = dict(st.secrets)

        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"):
            if k in secrets_dict:
                cleaned = _clean(secrets_dict[k])
                if cleaned:
                    return cleaned

        for _, sec_val in secrets_dict.items():
            if hasattr(sec_val, "items"):
                for sub_k, sub_v in sec_val.items():
                    if sub_k.lower() in ("gemini_api_key", "google_api_key", "api_key"):
                        cleaned = _clean(sub_v)
                        if cleaned:
                            return cleaned
    except Exception:
        pass
    return None


def build_gto_vsosh_verdict(chin_above, hang_sec, min_hang_sec, rep_errors):
    chin_tag = "Подбородок выше грифа: ✅ Да" if chin_above else "Подбородок выше грифа: ❌ Нет"
    hang_tag = f"Вис ИП: {hang_sec:.2f} с"

    tech_errors = [
        e for e in rep_errors
        if not e.startswith("фиксация ИП") and not e.startswith("подбородок ниже уровня грифа")
    ]

    if not chin_above:
        short_chin = [e for e in rep_errors if e.startswith("подбородок ниже уровня грифа")]
        reason = short_chin[0] if short_chin else "подбородок ниже грифа"
        verdict = f"❌ Не засчитано (ГТО/ВсОШ): {reason}"
    elif len(rep_errors) == 0:
        verdict = "✅ Засчитано (ГТО/ВсОШ): чистая техника и фиксация виса"
    elif hang_sec < min_hang_sec and not tech_errors:
        diff_h = round(min_hang_sec - hang_sec, 2)
        verdict = f"❌ Не засчитано (ГТО/ВсОШ): фиксация ИП {hang_sec:.2f} с < {min_hang_sec:.2f} с (не хватило: {max(0.01, diff_h):.2f} с)"
    else:
        short_e = "; ".join(tech_errors[:2]) if tech_errors else "; ".join(list(rep_errors)[:2])
        verdict = f"❌ Не засчитано (ГТО/ВсОШ): {short_e}"

    return f"{chin_tag} | {hang_tag} → {verdict}"


class GeminiDisputeVerifier:
    def __init__(self, enabled=True):
        self.api_key = _resolve_gemini_api_key()
        self.sdk = GEMINI_SDK
        self.enabled = bool(enabled and self.api_key)
        self.client = None
        self.init_error = None
        self.resolved_model = None

        if not enabled:
            self.init_error = "ИИ-агент отключен"
        elif not self.api_key:
            self.init_error = "ИИ-агент: ключ GEMINI_API_KEY не задан"
        else:
            try:
                self.resolved_model = self._discover_working_model()
                if self.sdk == "google-genai":
                    self.client = genai.Client(api_key=self.api_key)
                elif self.sdk == "legacy":
                    genai_legacy.configure(api_key=self.api_key)
            except Exception as e:
                self.init_error = f"Ошибка ИИ: {e}"

    def _discover_working_model(self):
        try:
            url = f"[https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){self.api_key}"
            req = urllib.request.Request(
                url,
                headers={"x-goog-api-key": self.api_key, "User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                for m in models:
                    methods = m.get("supportedGenerationMethods", [])
                    name = m.get("name", "").replace("models/", "")
                    if "generateContent" in methods and "flash" in name.lower() and "8b" not in name.lower():
                        return name
                for m in models:
                    methods = m.get("supportedGenerationMethods", [])
                    name = m.get("name", "").replace("models/", "")
                    if "generateContent" in methods and "gemini" in name.lower():
                        return name
        except Exception:
            pass
        return "gemini-1.5-flash"

    @staticmethod
    def build_storyboard_jpeg(start_frame, mid_frame, peak_frame, bar_y):
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

    def _call_gemini_rest_api(self, model_name, jpeg_bytes, prompt):
        clean_name = model_name.replace("models/", "")
        url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){clean_name}:generateContent?key={self.api_key}"
        b64_img = base64.b64encode(jpeg_bytes).decode("ascii")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "User-Agent": "Mozilla/5.0",
            },
        )

        proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        if proxy_url:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"https": proxy_url, "http": proxy_url})
            )
        else:
            opener = urllib.request.build_opener()

        with opener.open(req, timeout=20.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return ""

    def verify_independently(
        self,
        start_frame,
        mid_frame,
        peak_frame,
        bar_y,
        hang_sec=1.0,
        min_hang_sec=0.8,
        is_first_rep=False,
    ):
        jpeg_bytes = self.build_storyboard_jpeg(start_frame, mid_frame, peak_frame, bar_y)
        if not self.enabled or not jpeg_bytes:
            return None, "⚠️ ИИ-агент офлайн", False, jpeg_bytes

        prompt = (
            "Ты — верховный независимый судья соревнований по подтягиваниям на высокой перекладине (правила ГТО и ВсОШ).\n"
            "Перед тобой 3 ключевых стоп-кадра попытки:\n"
            "1. HANG (START) — исходный вис на перекладине.\n"
            "2. MID PULL — активная фаза подъема.\n"
            "3. PEAK (TOP) — верхняя точка подтягивания (желтая линия — верхний уровень грифа перекладины).\n"
            f"Зафиксированное время предварительной паузы в неподвижном висе: {hang_sec:.2f} сек (норматив: не менее {min_hang_sec:.1f} сек).\n\n"
            "Проведи НЕЗАВИСИМУЮ визуальную экспертизу строго по критериям:\n"
            "1. chin_above_bar: Поднялся ли подбородок спортсмена явно ВЫШЕ желтой линии грифа на кадре 3 (PEAK)?\n"
            "2. knees_straight: Ноги прямые в коленных суставах на всех фазах (не поджаты назад или к груди)?\n"
            "3. legs_clean: Ноги сведены вместе (стопы не скрещены и не разведены широко в стороны)?\n"
            "4. no_kipping: Подъем выполнен силой мышц рук и спины БЕЗ маха ногами, рывка тазом и раскачивания?\n"
            "5. arms_symmetric: Сгибание обеих рук равномерное и одновременное (нет резкого перекоса плеч или подтягивания сначала на одну руку)?\n\n"
            "Верни ТОЛЬКО валидный JSON со своим независимым вердиктом:\n"
            '{\n'
            '  "is_valid": bool,\n'
            '  "chin_above_bar": bool,\n'
            '  "knees_straight": bool,\n'
            '  "legs_clean": bool,\n'
            '  "no_kipping": bool,\n'
            '  "arms_symmetric": bool,\n'
            '  "errors": ["список выявленных ошибок на русском языке, либо пусто если попытка чистая"],\n'
            '  "judge_summary": "краткое экспертное заключение судьи до 10 слов"\n'
            '}'
        )

        models_to_try = []
        if self.resolved_model:
            models_to_try.append(self.resolved_model)
        for m in GEMINI_FALLBACK_MODELS:
            clean_m = m.replace("models/", "")
            if clean_m not in models_to_try:
                models_to_try.append(clean_m)

        raw_text = ""
        for clean_model in models_to_try:
            try:
                raw_text = self._call_gemini_rest_api(clean_model, jpeg_bytes, prompt)
                if raw_text.strip():
                    break
            except Exception:
                continue

        if not raw_text.strip():
            return None, "⚠️ Нет ответа от ИИ-агента", False, jpeg_bytes

        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").replace("json", "", 1).strip()

            v = json.loads(cleaned)
            ai_errors = list(v.get("errors", []))
            chin_ok = bool(v.get("chin_above_bar", True))
            knees_ok = bool(v.get("knees_straight", True))
            legs_ok = bool(v.get("legs_clean", True))
            kip_ok = bool(v.get("no_kipping", True))
            arms_ok = bool(v.get("arms_symmetric", True))

            if not is_first_rep and hang_sec < min_hang_sec:
                hang_err = f"фиксация ИП менее {min_hang_sec:.1f} сек ({hang_sec:.2f} с)"
                if hang_err not in ai_errors:
                    ai_errors.append(hang_err)

            if not chin_ok and "подбородок ниже уровня грифа перекладины" not in ai_errors:
                ai_errors.append("подбородок ниже уровня грифа перекладины")
            if not knees_ok and "нарушение ИП: согнутые в коленях ноги" not in ai_errors:
                ai_errors.append("нарушение ИП: согнутые в коленях ноги")
            if not legs_ok and "нарушение ИП: ноги не вместе / скрещены" not in ai_errors:
                ai_errors.append("нарушение ИП: ноги не вместе / скрещены")
            if not kip_ok and "подтягивание рывками или с махом ногами/туловищем" not in ai_errors:
                ai_errors.append("подтягивание рывками или с махом ногами/туловищем")
            if not arms_ok and "поочередное (неравномерное) сгибание рук" not in ai_errors:
                ai_errors.append("поочередное (неравномерное) сгибание рук")

            is_valid = (len(ai_errors) == 0) and (hang_sec >= min_hang_sec or is_first_rep)
            chin_str = "✅ Да" if chin_ok else "❌ Нет"
            summary = v.get("judge_summary", "").strip() or ("Чистое выполнение" if is_valid else ", ".join(ai_errors[:2]))
            note = f"ИИ: {'✅ Засчитано' if is_valid else '❌ Не засчитано'} (Подбородок: {chin_str} | ИП: {hang_sec:.2f} с) — {summary}"

            return set(ai_errors), note, True, jpeg_bytes
        except Exception as e:
            return None, f"⚠️ Ошибка разбора ответа ИИ: {e}", False, jpeg_bytes


def get_yolo_model():
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

        if has_onnxruntime and os.path.exists(onnx_file):
            _MODEL_INSTANCE = YOLO(onnx_file, task="pose")
        else:
            _MODEL_INSTANCE = YOLO(YOLO_MODEL_NAME)
    return _MODEL_INSTANCE


def select_target_athlete(yolo_result, prev_center=None, locked_bar_y=None):
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
        if locked_bar_y is not None:
            hands_up_bonus = 0.75 if abs(wrists_y - locked_bar_y) < 0.14 else -0.60
        else:
            hands_up_bonus = 0.50 if wrists_y < cy else -0.45

        center_dist = math.hypot(cx - 0.5, cy - 0.45)

        if prev_center is not None:
            track_dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
            if track_dist > 0.20:
                continue
            score = -4.5 * track_dist + 1.8 * area + hands_up_bonus
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
    HAND_INDICES = {LM.LEFT_WRIST, LM.RIGHT_WRIST}

    def __init__(self, config=CONFIG):
        self.cfg = config
        self.filters_x = {}
        self.filters_y = {}
        self.prev_pts = None

        self.orientation_sign = None
        self.sh_w_history = deque(maxlen=30)
        self.hip_w_history = deque(maxlen=30)
        self.wrist_w_history = deque(maxlen=30)

        self.ref_sh_w = None
        self.ref_hip_w = None
        self.ref_wrist_w = None

        self.anchor_l_wrist = None
        self.anchor_r_wrist = None
        self.hang_hip_y = None
        self.elbow_outward_ratio = 0.08

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
                center_x = 0.70 * prev_center_x + 0.30 * raw_center_x
            else:
                center_x = 0.5 * (lx + rx)

            sign = self.orientation_sign if self.orientation_sign is not None else (1.0 if curr_dx >= 0 else -1.0)
            restored_w = 0.85 * target_width + 0.15 * max(curr_w, min_width)
            half_w = 0.5 * restored_w
            pts[left_idx].x = center_x + sign * half_w
            pts[right_idx].x = center_x - sign * half_w

            if self.prev_pts is not None:
                pts[left_idx].y = 0.70 * self.prev_pts[left_idx].y + 0.30 * pts[left_idx].y
                pts[right_idx].y = 0.70 * self.prev_pts[right_idx].y + 0.30 * pts[right_idx].y
            return True
        return False

    def process(self, raw_landmarks, timestamp_sec, bar_y=None):
        pts = [
            StabilizedLandmark(lm.x, lm.y, lm.z, lm.visibility, lm.presence)
            for lm in raw_landmarks
        ]

        sh_y = 0.5 * (pts[LM.LEFT_SHOULDER].y + pts[LM.RIGHT_SHOULDER].y)
        hip_y = 0.5 * (pts[LM.LEFT_HIP].y + pts[LM.RIGHT_HIP].y)

        if self.hang_hip_y is None and hip_y > 0.25:
            self.hang_hip_y = hip_y
        elif self.hang_hip_y is not None and hip_y > self.hang_hip_y:
            self.hang_hip_y = 0.92 * self.hang_hip_y + 0.08 * hip_y

        athlete_suspended = (
            self.hang_hip_y is None
            or (hip_y <= self.hang_hip_y + 0.10 and sh_y <= (bar_y or 0.18) + 0.28)
        )

        l_wr, r_wr = pts[LM.LEFT_WRIST], pts[LM.RIGHT_WRIST]
        wrists_near_bar = (
            bar_y is not None
            and abs(l_wr.y - bar_y) < 0.075
            and abs(r_wr.y - bar_y) < 0.075
            and abs(l_wr.x - r_wr.x) > 0.065
        )

        if wrists_near_bar and l_wr.visibility >= 0.45 and r_wr.visibility >= 0.45:
            if self.anchor_l_wrist is None:
                self.anchor_l_wrist = [l_wr.x, l_wr.y]
                self.anchor_r_wrist = [r_wr.x, r_wr.y]
            else:
                self.anchor_l_wrist[0] = 0.90 * self.anchor_l_wrist[0] + 0.10 * l_wr.x
                self.anchor_l_wrist[1] = 0.90 * self.anchor_l_wrist[1] + 0.10 * l_wr.y
                self.anchor_r_wrist[0] = 0.90 * self.anchor_r_wrist[0] + 0.10 * r_wr.x
                self.anchor_r_wrist[1] = 0.90 * self.anchor_r_wrist[1] + 0.10 * r_wr.y

        if athlete_suspended and self.anchor_l_wrist is not None and self.anchor_r_wrist is not None:
            max_dy = self.cfg["wrist_lock_max_drift_y"]
            max_dx = self.cfg["wrist_lock_max_drift_x"]

            if abs(l_wr.y - self.anchor_l_wrist[1]) > max_dy or abs(l_wr.x - self.anchor_l_wrist[0]) > max_dx or l_wr.visibility < 0.35:
                l_wr.x, l_wr.y, l_wr.visibility = self.anchor_l_wrist[0], self.anchor_l_wrist[1], 0.95

            if abs(r_wr.y - self.anchor_r_wrist[1]) > max_dy or abs(r_wr.x - self.anchor_r_wrist[0]) > max_dx or r_wr.visibility < 0.35:
                r_wr.x, r_wr.y, r_wr.visibility = self.anchor_r_wrist[0], self.anchor_r_wrist[1], 0.95

        if self.prev_pts is not None:
            max_jump = self.cfg["max_joint_jump_per_frame"]
            for idx, p in enumerate(pts):
                if athlete_suspended and idx in self.HAND_INDICES and self.anchor_l_wrist is not None:
                    continue
                prev_p = self.prev_pts[idx]
                conf = p.visibility
                if (p.x <= 1e-4 and p.y <= 1e-4) or conf < 0.25:
                    p.x, p.y = prev_p.x, prev_p.y
                    continue

                dist = math.hypot(p.x - prev_p.x, p.y - prev_p.y)
                if conf < 0.45:
                    p.x = prev_p.x + 0.20 * (p.x - prev_p.x)
                    p.y = prev_p.y + 0.20 * (p.y - prev_p.y)
                elif dist > max_jump:
                    scale = max_jump / dist
                    p.x = prev_p.x + (p.x - prev_p.x) * scale
                    p.y = prev_p.y + (p.y - prev_p.y) * scale

        wrist_y = 0.5 * (pts[LM.LEFT_WRIST].y + pts[LM.RIGHT_WRIST].y)
        hands_on_bar = athlete_suspended and ((bar_y is None) or (abs(wrist_y - bar_y) < self.cfg["bar_release_dist"]))

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
                    min_width=self.ref_wrist_w * 0.85,
                    target_width=self.ref_wrist_w,
                )

            self._restore_pair_width(
                pts, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
                min_width=self.ref_sh_w * collapse_ratio,
                target_width=self.ref_sh_w,
            )

            target_hip_w = self.ref_hip_w if self.ref_hip_w is not None else (self.ref_sh_w * 0.62)
            self._restore_pair_width(
                pts, LM.LEFT_HIP, LM.RIGHT_HIP,
                min_width=target_hip_w * collapse_ratio,
                target_width=target_hip_w,
            )

            for sh_i, el_i, wr_i, side_sign in (
                (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, 1.0),
                (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, -1.0),
            ):
                sh_pt, el_pt, wr_pt = pts[sh_i], pts[el_i], pts[wr_i]
                arm_span_y = sh_pt.y - wr_pt.y
                if arm_span_y > 0.085:
                    expected_el_y = 0.50 * sh_pt.y + 0.50 * wr_pt.y
                    if el_pt.y > sh_pt.y - 0.010 or el_pt.y < wr_pt.y + 0.005 or abs(el_pt.y - expected_el_y) > 0.045:
                        el_pt.y = expected_el_y
                        sign = (self.orientation_sign or 1.0) * side_sign
                        el_pt.x = 0.52 * sh_pt.x + 0.48 * wr_pt.x + sign * (0.04 * self.ref_sh_w)

            el_dx = pts[LM.LEFT_ELBOW].x - pts[LM.RIGHT_ELBOW].x
            el_w = abs(el_dx)
            expected_arm_span = 0.5 * (
                abs(pts[LM.LEFT_SHOULDER].x - pts[LM.RIGHT_SHOULDER].x)
                + abs(pts[LM.LEFT_WRIST].x - pts[LM.RIGHT_WRIST].x)
            )
            min_el_w = max(self.ref_sh_w * collapse_ratio, expected_arm_span * 0.68)

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

        smoothed_pts = []
        for idx, p in enumerate(pts):
            fx, fy = self._get_filters(idx)
            if hands_on_bar and idx in self.HAND_INDICES:
                sx = fx.filter(p.x, timestamp_sec, min_cutoff_override=0.10, beta_override=0.002)
                sy = fy.filter(p.y, timestamp_sec, min_cutoff_override=0.10, beta_override=0.002)
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
    l_wr, r_wr = lm[LM.LEFT_WRIST], lm[LM.RIGHT_WRIST]
    if l_wr.visibility >= 0.35 and r_wr.visibility >= 0.35:
        wrist_y = 0.5 * (l_wr.y + r_wr.y)
    elif l_wr.visibility > r_wr.visibility:
        wrist_y = l_wr.y
    else:
        wrist_y = r_wr.y

    return float(np.clip(wrist_y - 0.004, 0.05, 0.35))


def compute_anatomical_chin(lm, base_shoulder_w=0.12):
    shoulder_y = 0.5 * (lm[LM.LEFT_SHOULDER].y + lm[LM.RIGHT_SHOULDER].y)
    shoulder_x = 0.5 * (lm[LM.LEFT_SHOULDER].x + lm[LM.RIGHT_SHOULDER].x)

    nose_lm = lm[LM.NOSE]
    nose_visible = (nose_lm.visibility >= 0.35 and nose_lm.y > 0.015)

    if nose_visible:
        nose_x, nose_y = nose_lm.x, nose_lm.y
        l_eye, r_eye = lm[LM.LEFT_EYE], lm[LM.RIGHT_EYE]
        if l_eye.visibility > 0.25 and r_eye.visibility > 0.25 and l_eye.y > 0 and r_eye.y > 0:
            eye_y = 0.5 * (l_eye.y + r_eye.y)
            eye_to_nose = max(0.008, nose_y - eye_y)
            chin_drop = np.clip(1.15 * eye_to_nose, 0.15 * base_shoulder_w, 0.26 * base_shoulder_w)
        else:
            chin_drop = 0.18 * base_shoulder_w

        chin_y = float(np.clip(nose_y + chin_drop, nose_y + 0.008, shoulder_y - 0.012))
        chin_x = 0.80 * nose_x + 0.20 * shoulder_x
    else:
        neck_height = 0.24 * base_shoulder_w
        chin_y = float(shoulder_y - neck_height)
        chin_x = float(shoulder_x)

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
    h, w, _ = frame.shape
    if bar_y is not None:
        bar_px = int(bar_y * h)
        bar_color = (0, 255, 0) if chin_above_now else (0, 215, 255)
        cv2.line(frame, (0, bar_px), (w, bar_px), bar_color, 3)
        label_y = bar_px + 22 if bar_px < 35 else bar_px - 8
        rep_tag = f"REP #{current_rep_idx}" if state != "WAIT_HANG" else f"NEXT REP #{current_rep_idx}"
        chin_tag = "[CHIN ABOVE: YES]" if chin_above_now else ""
        label_text = f"BAR LINE [{rep_tag}: Y={bar_y:.3f}] {chin_tag}"
        cv2.putText(frame, label_text, (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
        cv2.putText(frame, label_text, (15, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, bar_color, 2)

    if ai_banner_active and ai_last_rep_num is not None:
        banner_overlay = frame.copy()
        bw, bh = min(w - 20, 410), 58
        bx1 = max(10, w - bw - 12)
        by1 = 12
        bg_color = (25, 95, 25) if ai_last_valid else (45, 20, 110)
        border_color = (0, 255, 120) if ai_last_valid else (255, 80, 255)

        cv2.rectangle(banner_overlay, (bx1, by1), (bx1 + bw, by1 + bh), bg_color, -1)
        cv2.addWeighted(banner_overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx1 + bw, by1 + bh), border_color, 2)

        verdict_tag = "VALID (GTO/VSOSH)" if ai_last_valid else "REJECTED (GTO/VSOSH)"
        cv2.putText(
            frame,
            f"VERDICT REP #{ai_last_rep_num}",
            (bx1 + 12, by1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{verdict_tag}",
            (bx1 + 12, by1 + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    overlay = frame.copy()
    box_w, box_h = min(w - 20, 460), 158
    y_top = max(10, h - box_h - 15)
    cv2.rectangle(overlay, (10, y_top), (10 + box_w, y_top + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    cv2.putText(frame, f"GTO/VSOSH VALID: {valid_reps} / ATTEMPTS: {total_attempts}", (20, y_top + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 120), 2)
    cv2.putText(frame, f"STATE: {state} | HANG PAUSE: {hang_sec:.2f}s", (20, y_top + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"ELBOW: {int(elbow_deg)} deg | KNEE: {int(knee_deg)} deg", (20, y_top + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 2)

    err_status = "TECHNIQUE: VIOLATION" if active_errors else "TECHNIQUE: CLEAN (OK)"
    err_color = (0, 80, 255) if active_errors else (0, 255, 0)
    cv2.putText(frame, err_status, (20, y_top + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, err_color, 2)

    chin_hud_txt = "CHIN ABOVE BAR: YES" if chin_above_now else "CHIN ABOVE BAR: NO"
    chin_hud_col = (0, 255, 120) if chin_above_now else (180, 180, 180)
    cv2.putText(frame, chin_hud_txt, (20, y_top + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.54, chin_hud_col, 2, cv2.LINE_AA)


def get_calibration_preview(video_path, manual_bar_y=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, 0.155

    h, w, _ = frame.shape
    detected_bar_y = 0.155

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
    cv2.line(frame, (0, bar_px), (w, bar_px), bar_color := (0, 255, 0), 3)
    cv2.putText(frame, f"Bar Line Y={bar_y:.3f}", (20, max(30, bar_px - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round(detected_bar_y, 3)


def convert_to_web_mp4(raw_video_path):
    if not os.path.exists(raw_video_path) or os.path.getsize(raw_video_path) < 1000:
        return raw_video_path

    web_path = raw_video_path.replace(".mp4", "_web.mp4")

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe and os.path.exists("ffmpeg.exe"):
        ffmpeg_exe = os.path.abspath("ffmpeg.exe")
    if not ffmpeg_exe:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = None

    if ffmpeg_exe:
        try:
            cmd = [
                ffmpeg_exe, "-y", "-i", raw_video_path,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "ultrafast", "-crf", "28",
                "-movflags", "+faststart", web_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
            if res.returncode == 0 and os.path.exists(web_path) and os.path.getsize(web_path) > 1000:
                try:
                    os.remove(raw_video_path)
                except OSError:
                    pass
                return web_path
        except Exception:
            pass

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
        "bar_y": round(manual_bar_y if manual_bar_y is not None else 0.155, 3),
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
    bar_y = manual_bar_y if manual_bar_y is not None else 0.155
    bar_is_locked = (manual_bar_y is not None)

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

        frame_step = max(1, int(round(fps / CONFIG["target_eval_fps"])))
        eff_fps = fps / frame_step
        fps_scale = eff_fps / 30.0

        min_asym_f = max(2, int(round(CONFIG["asym_min_frames"] * fps_scale)))
        min_knee_f = max(2, int(round(CONFIG["knee_err_min_frames"] * fps_scale)))
        min_cross_f = max(3, int(round(CONFIG["cross_err_min_frames"] * fps_scale)))
        min_feet_f = max(3, int(round(CONFIG["feet_apart_min_frames"] * fps_scale)))
        min_kip_f = max(2, int(round(CONFIG["kipping_min_frames"] * fps_scale)))

        scale = min(1.0, 720.0 / max(orig_w, orig_h, 1))
        out_w = max(2, int(orig_w * scale) // 2 * 2)
        out_h = max(2, int(orig_h * scale) // 2 * 2)

        if generate_tracked_video:
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            raw_out_path = tmp_out.name
            tmp_out.close()
            for codec_name in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_w = cv2.VideoWriter(raw_out_path, fourcc, eff_fps, (out_w, out_h))
                if test_w.isOpened():
                    writer = test_w
                    break
                test_w.release()

        state = "WAIT_HANG"
        hang_start_time = None
        last_hang_duration = 0.0
        pre_bend_frames = 0
        hands_lost_frames = 0

        initial_bar_samples = []
        shoulder_w_samples = []
        wrist_w_samples = []
        base_sh_tilt_samples = []
        base_el_tilt_samples = []
        base_el_diff_samples = []

        base_shoulder_w = 0.12
        base_wrist_w = 0.16
        base_sh_tilt = 0.03
        base_el_tilt = 0.03
        base_el_diff = 5.0
        hang_shoulder_y_ref = 0.30

        stabilizer = SkeletonStabilizer(CONFIG)
        verifier = GeminiDisputeVerifier(enabled=(CONFIG["enable_gemini_verifier"] or force_ai_all_reps))

        chin_x_filter = OneEuroFilter(
            min_cutoff=CONFIG["filter_min_cutoff_x"],
            beta=CONFIG["filter_beta_x"]
        )
        chin_y_filter = OneEuroFilter(
            min_cutoff=CONFIG["filter_min_cutoff_y"],
            beta=CONFIG["filter_beta_y"]
        )

        buf_len = 2 if eff_fps <= 18.0 else 3
        chin_buf = deque(maxlen=buf_len)
        nose_buf = deque(maxlen=buf_len)
        elbow_buf = deque(maxlen=buf_len)
        knee_buf = deque(maxlen=buf_len)
        shoulder_y_buf = deque(maxlen=buf_len)

        pre_pull_window = max(6, int(round(0.7 * eff_fps)))
        pre_pull_knee_err = deque(maxlen=pre_pull_window)
        pre_pull_cross_err = deque(maxlen=pre_pull_window)

        rep_start_time = 0.0
        rep_saved_hang_time = None
        rep_hang_pause_sec = 0.0
        rep_start_shoulder_y = 0.30
        min_elbow_in_rep = 180.0
        min_shoulder_y_in_rep = 1.0
        min_chin_y_in_rep = 1.0
        min_knee_in_rep = 180.0
        min_hip_in_rep = 180.0
        max_el_diff_in_rep = 0.0
        max_feet_apart_ratio = 0.0
        max_ear_drop_in_rep = -1.0
        max_elbow_in_descent = 0.0
        max_shoulder_y_in_descent = 0.0
        descent_bottom_Start_time = None
        chin_above_frames = 0
        head_thrown_back_frames = 0
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

        def _finalize_and_record_rep(end_time_sec):
            nonlocal ai_calls_count, ai_last_trigger_time, ai_last_rep_num, ai_last_valid
            ankle_swing_span = (max(rep_ankle_x) - min(rep_ankle_x)) if rep_ankle_x else 0.0
            actual_swing_ratio = round(ankle_swing_span / max(0.06, base_shoulder_w), 2)

            asym_thresh = max(min_asym_f, int(rep_frames * CONFIG["asym_frame_ratio"]))
            knee_thresh = max(min_knee_f, int(rep_frames * CONFIG["knee_err_frame_ratio"]))
            cross_thresh = min_cross_f
            feet_thresh = max(min_feet_f, int(rep_frames * CONFIG["feet_apart_frame_ratio"]))
            kip_thresh = max(min_kip_f, int(rep_frames * CONFIG["kipping_frame_ratio"]))

            # 1. Асимметрия сгибания рук с точным перекосом
            if asym_frames >= asym_thresh:
                norm_asym = int(CONFIG["asym_elbow_diff_deg"])
                delta_asym = int(max_el_diff_in_rep - norm_asym)
                rep_errors.add(
                    f"поочередное сгибание рук (разница локтей: {int(max_el_diff_in_rep)}° при норме ≤ {norm_asym}°, перекос: +{max(1, delta_asym)}°)"
                )

            # 2. Согнутые колени с точным отклонением угла
            if knee_bent_frames >= knee_thresh:
                norm_knee = int(CONFIG["knee_bent_threshold"])
                delta_knee = int(norm_knee - min_knee_in_rep)
                rep_errors.add(
                    f"нарушение ИП: согнутые ноги (угол колен: {int(min_knee_in_rep)}° при норме ≥ {norm_knee}°, отклонение: -{max(1, delta_knee)}°)"
                )

            # 3. Перекрещенные ноги
            if legs_crossed_frames >= cross_thresh:
                rep_errors.add("нарушение ИП: перекрещенные ноги в лодыжках/голенях")

            # 4. Разведение стоп с точным коэффициентом ширины плеч
            if feet_apart_frames >= feet_thresh:
                norm_feet = CONFIG["feet_apart_ratio"]
                delta_feet = round(max_feet_apart_ratio - norm_feet, 2)
                rep_errors.add(
                    f"нарушение ИП: ступни не вместе (разведение: {max_feet_apart_ratio:.2f} шир. плеч при норме ≤ {norm_feet:.2f}, превышение: +{max(0.01, delta_feet):.2f})"
                )

            # 5. Киппинг и мах ногами с точным размахом
            if (
                kipping_frames >= kip_thresh
                or actual_swing_ratio > CONFIG["ankle_swing_span_ratio"]
            ):
                norm_swing = CONFIG["ankle_swing_span_ratio"]
                delta_swing = round(actual_swing_ratio - norm_swing, 2)
                rep_errors.add(
                    f"подтягивание рывками/махом (размах маха: {actual_swing_ratio:.2f} шир. плеч при норме ≤ {norm_swing:.2f}, превышение: +{max(0.01, delta_swing):.2f}; таз: {int(min_hip_in_rep)}°)"
                )

            # 6. Закидывание головы назад с точной дельтой смещения
            if head_thrown_back_frames >= CONFIG.get("head_thrown_back_min_frames", 4):
                norm_head = CONFIG.get("head_thrown_back_ear_drop", 0.042)
                delta_head = round(max_ear_drop_in_rep - norm_head, 3)
                rep_errors.add(
                    f"нарушение техники: закидывание головы назад (смещение ушей к носу: {max_ear_drop_in_rep:.3f} при норме ≤ {norm_head:.3f}, превышение: +{max(0.001, delta_head):.3f})"
                )

            is_wide_grip = (base_wrist_w / max(0.06, base_shoulder_w)) > 1.55
            eff_max_elbow = CONFIG["wide_grip_max_elbow"] if is_wide_grip else CONFIG["narrow_grip_max_elbow"]

            # Строгая локальная фиксация: подбородок обязан фактически пересечь гриф
            chin_was_above_local = (
                chin_above_frames >= CONFIG["min_chin_above_frames"]
                or (
                    min_chin_y_in_rep <= bar_y + CONFIG["chin_bar_tolerance"]
                    and min_elbow_in_rep <= eff_max_elbow
                )
            )
            local_errors = set(rep_errors)
            if not chin_was_above_local:
                cm_diff = round((min_chin_y_in_rep - bar_y) / max(0.06, base_shoulder_w) * 40.0, 1)
                cm_str = f"{abs(cm_diff):.1f} см ниже грифа" if cm_diff > 0 else "нет фиксации над перекладиной"
                local_errors.add(
                    f"подбородок ниже уровня грифа ({cm_str}, угол локтей: {int(min_elbow_in_rep)}° при норме ≤ {int(eff_max_elbow)}°)"
                )

            storyboard_jpeg_bytes = verifier.build_storyboard_jpeg(
                rep_start_frame, rep_mid_frame, rep_peak_frame, bar_y
            )

            # ПОЛНОСТЬЮ НЕЗАВИСИМАЯ ЭКСПЕРТИЗА ИИ-АГЕНТА
            ai_was_triggered = False
            ai_errors = None
            ai_note = "—"
            ai_indicator_str = "⚖️ Локальный FSM"
            is_first_rep = (len(reps_log) == 0 and rep_start_time < 2.2)

            if verifier.enabled and storyboard_jpeg_bytes is not None:
                ai_errs, ai_verdict_text, ai_ok, sb_bytes = verifier.verify_independently(
                    start_frame=rep_start_frame,
                    mid_frame=rep_mid_frame,
                    peak_frame=rep_peak_frame,
                    bar_y=bar_y,
                    hang_sec=rep_hang_pause_sec,
                    min_hang_sec=min_hang_sec,
                    is_first_rep=is_first_rep,
                )
                if sb_bytes:
                    storyboard_jpeg_bytes = sb_bytes
                if ai_ok and ai_errs is not None:
                    ai_was_triggered = True
                    ai_errors = ai_errs
                    ai_note = ai_verdict_text
                    ai_indicator_str = "🤖 Независимый ИИ-судья"

            # СУВЕРЕННОЕ РЕШЕНИЕ: приоритет отдается независимому ИИ-агенту
            if ai_was_triggered and ai_errors is not None:
                final_errors = ai_errors
                chin_was_above = ("подбородок ниже уровня грифа перекладины" not in final_errors)
            else:
                final_errors = local_errors
                chin_was_above = chin_was_above_local
                ai_note = build_gto_vsosh_verdict(chin_was_above, rep_hang_pause_sec, min_hang_sec, final_errors)

            is_valid = (len(final_errors) == 0)
            current_rep_num = len(reps_log) + 1

            # Синхронизируем баннер вердикта для локального и облачного режима
            ai_calls_count += 1
            ai_last_trigger_time = end_time_sec
            ai_last_rep_num = current_rep_num
            ai_last_valid = is_valid

            status_str = "✅ Засчитано" if is_valid else "❌ Не засчитано"
            b64_img = base64.b64encode(storyboard_jpeg_bytes).decode("ascii") if storyboard_jpeg_bytes else None

            reps_log.append({
                "rep_number": current_rep_num,
                "timestamp": f"{format_timestamp(rep_start_time)} – {format_timestamp(end_time_sec)}",
                "hang_pause_sec": round(rep_hang_pause_sec, 2),
                "chin_above_bar": "✅ Да" if chin_was_above else "❌ Нет",
                "bar_y": round(bar_y, 3),
                "valid": is_valid,
                "status_text": status_str,
                "verdict_gto": "✅ Засчитано (ГТО/ВсОШ)" if is_valid else "❌ Не засчитано (ГТО/ВсОШ)",
                "ai_triggered": ai_was_triggered,
                "ai_indicator": ai_indicator_str,
                "ai_verdict": ai_note,
                "ai_storyboard_b64": b64_img,
                "errors": sorted(final_errors) if final_errors else ["Нет ошибок (чистое выполнение)"]
            })

        frame_idx = 0
        eval_idx = 0
        tracked_athlete_center = None
        model = get_yolo_model()
        yolo_imgsz = CONFIG["yolo_imgsz"]

        while cap.isOpened():
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

            preds = model.predict(frame, conf=0.35, imgsz=yolo_imgsz, verbose=False)
            raw_lm, tracked_athlete_center = (
                select_target_athlete(preds[0], tracked_athlete_center, locked_bar_y=(bar_y if bar_is_locked else None))
                if preds
                else (None, tracked_athlete_center)
            )

            mean_elbow_angle = 0.0
            mean_knee_angle = 180.0
            chin_above_now = False

            if raw_lm is not None:
                if not bar_is_locked:
                    instant_bar_y = compute_bar_y_from_hands(raw_lm, ref_sh_w=base_shoulder_w)
                    if 0.08 <= instant_bar_y <= 0.25:
                        initial_bar_samples.append(instant_bar_y)
                        bar_y = float(np.median(initial_bar_samples))
                        if len(initial_bar_samples) >= CONFIG["bar_calibration_frames"]:
                            bar_is_locked = True

                lm = stabilizer.process(raw_lm, current_time, bar_y=bar_y)

                def pt(idx):
                    return [lm[idx].x, lm[idx].y]

                l_sh, r_sh = pt(LM.LEFT_SHOULDER), pt(LM.RIGHT_SHOULDER)
                l_el, r_el = pt(LM.LEFT_ELBOW), pt(LM.RIGHT_ELBOW)
                l_wr, r_wr = pt(LM.LEFT_WRIST), pt(LM.RIGHT_WRIST)
                l_hip, r_hip = pt(LM.LEFT_HIP), pt(LM.RIGHT_HIP)
                l_kn, r_kn = pt(LM.LEFT_KNEE), pt(LM.RIGHT_KNEE)
                l_an, r_an = pt(LM.LEFT_ANKLE), pt(LM.RIGHT_ANKLE)
                l_eye, r_eye = pt(LM.LEFT_EYE), pt(LM.RIGHT_EYE)
                l_ear, r_ear = pt(LM.LEFT_EAR), pt(LM.RIGHT_EAR)

                wrist_y = (l_wr[1] + r_wr[1]) / 2.0
                nose = pt(LM.NOSE)
                eye_y = (l_eye[1] + r_eye[1]) / 2.0
                ear_y = (l_ear[1] + r_ear[1]) / 2.0
                raw_chin_x, raw_chin_y = compute_anatomical_chin(lm, base_shoulder_w)

                chin_x = chin_x_filter.filter(raw_chin_x, current_time)
                filtered_chin_y = chin_y_filter.filter(raw_chin_y, current_time)
                raw_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
                shoulder_center_x = (l_sh[0] + r_sh[0]) / 2.0

                angle_el_l = calculate_angle(l_sh, l_el, l_wr)
                angle_el_r = calculate_angle(r_sh, r_el, r_wr)
                raw_elbow = (angle_el_l + angle_el_r) / 2.0

                l_an_conf = lm[LM.LEFT_ANKLE].visibility
                r_an_conf = lm[LM.RIGHT_ANKLE].visibility
                ankles_lost_or_hidden = (l_an_conf < 0.38 and r_an_conf < 0.38)

                angle_kn_2d = (calculate_angle(l_hip, l_kn, l_an) + calculate_angle(r_hip, r_kn, r_an)) / 2.0
                thigh_len = (np.hypot(l_kn[0] - l_hip[0], l_kn[1] - l_hip[1]) + np.hypot(r_kn[0] - r_hip[0], r_kn[1] - r_hip[1])) / 2.0
                shin_len = (np.hypot(l_an[0] - l_kn[0], l_an[1] - l_kn[1]) + np.hypot(r_an[0] - r_kn[0], r_an[1] - r_kn[1])) / 2.0
                shin_ratio = shin_len / max(0.04, thigh_len)

                if ankles_lost_or_hidden:
                    raw_knee = 135.0
                elif shin_ratio < CONFIG["shin_ratio_threshold"]:
                    sagittal_knee_deg = float(np.degrees(np.arcsin(np.clip(shin_ratio / 0.85, 0.1, 1.0))) * 1.6)
                    raw_knee = min(angle_kn_2d, sagittal_knee_deg)
                else:
                    raw_knee = angle_kn_2d

                chin_buf.append(filtered_chin_y)
                nose_buf.append(nose[1])
                elbow_buf.append(raw_elbow)
                knee_buf.append(raw_knee)
                shoulder_y_buf.append(raw_shoulder_y)

                nose_y = float(min(nose[1], np.median(nose_buf)))
                shoulder_y = float(np.median(shoulder_y_buf))
                peak_chin_candidate = min(raw_chin_y, filtered_chin_y, float(np.median(chin_buf)))
                chin_y = float(np.clip(
                    peak_chin_candidate,
                    nose_y + 0.008,
                    shoulder_y - 0.012
                ))
                mean_elbow_angle = float(min(raw_elbow, np.median(elbow_buf)))
                mean_knee_angle = float(np.median(knee_buf))

                mean_hip_angle = (calculate_angle(l_sh, l_hip, l_kn) + calculate_angle(r_sh, r_hip, r_kn)) / 2.0
                curr_sh_w = abs(l_sh[0] - r_sh[0])
                curr_wr_w = abs(l_wr[0] - r_wr[0])

                if mean_elbow_angle >= 155.0 and curr_sh_w > 0.05 and len(shoulder_w_samples) < 30:
                    shoulder_w_samples.append(curr_sh_w)
                    wrist_w_samples.append(curr_wr_w)
                    base_shoulder_w = float(np.median(shoulder_w_samples))
                    base_wrist_w = float(np.median(wrist_w_samples))
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

                # Адаптивный порог асимметрии с учетом бокового солнца
                asym_diff_thresh = max(CONFIG["asym_elbow_diff_deg"], base_el_diff + 14.0)
                head_fully_in_frame = (nose[1] > 0.015)
                is_asym_frame = (
                    head_fully_in_frame
                    and (25.0 < mean_elbow_angle < 152.0)
                    and (
                        net_el_diff > asym_diff_thresh
                        or (
                            net_el_diff > CONFIG["asym_combo_elbow_deg"]
                            and (
                                net_sh_tilt > CONFIG["asym_shoulder_tilt_ratio"]
                                or net_el_tilt > CONFIG["asym_elbow_tilt_ratio"]
                            )
                        )
                        or (
                            net_sh_tilt > CONFIG["asym_shoulder_tilt_ratio"] * 1.25
                            and net_el_tilt > CONFIG["asym_elbow_tilt_ratio"] * 1.25
                        )
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
                        and ankle_dx * knee_dx < CONFIG["cross_product_threshold"]
                    )
                    and (ankle_dist > hip_w * CONFIG["cross_min_overlap_ratio"])
                )
                ankles_stacked_vertically = (
                    knee_dist > hip_w * 0.62
                    and ankle_dist < knee_dist * 0.30
                    and ankle_dy > hip_w * 0.35
                )
                legs_crossed = shins_crossed_x or ankles_stacked_vertically

                feet_apart = ankle_dist > (base_shoulder_w * CONFIG["feet_apart_ratio"])
                knees_bent = (
                    (mean_knee_angle < CONFIG["knee_bent_threshold"])
                    or (shin_ratio < CONFIG["shin_ratio_threshold"])
                    or ankles_lost_or_hidden
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

                # Детекция фронтального подброса коленей к груди
                thigh_foreshortened = (
                    state != "WAIT_HANG"
                    and (thigh_len < shin_len * CONFIG["thigh_foreshorten_ratio"])
                )

                kipping_detected = (
                    (mean_hip_angle < CONFIG["hip_kipping_angle"])
                    or (center_body_angle < CONFIG["center_body_bend_angle"])
                    or horizontal_swing
                    or thigh_foreshortened
                )

                # Откалиброванная детекция реального запрокидывания головы:
                # лицо обращено в потолок (глаза на одном уровне с носом или ниже)
                # плюс уши значительно опущены относительно носа при неполном сгибе рук
                is_head_thrown_back = (
                    state != "WAIT_HANG"
                    and (eye_y >= nose_y - 0.004)
                    and (ear_y > nose_y + CONFIG.get("head_thrown_back_ear_drop", 0.042))
                    and (mean_elbow_angle > 68.0)
                )

                hands_now_on_bar = abs(wrist_y - bar_y) < CONFIG["bar_release_dist"]
                if hands_now_on_bar:
                    hands_lost_frames = 0
                else:
                    hands_lost_frames += 1

                hands_on_bar = (hands_lost_frames < CONFIG["release_confirm_frames"])

                # Адаптивные лимиты локтей и плеч
                is_wide_grip = (base_wrist_w / max(0.06, base_shoulder_w)) > 1.55
                eff_max_elbow = CONFIG["wide_grip_max_elbow"] if is_wide_grip else CONFIG["narrow_grip_max_elbow"]
                eff_shoulder_drop = CONFIG["wide_grip_shoulder_drop"] if is_wide_grip else CONFIG["narrow_grip_shoulder_drop"]

                # Фиксация подбородка независима от угла наклона головы
                chin_reached = (
                    chin_y <= bar_y + CONFIG["chin_bar_tolerance"]
                    or (nose_y <= bar_y + CONFIG["nose_bar_margin"] and shoulder_y <= bar_y + eff_shoulder_drop)
                )
                chin_above_now = (
                    chin_reached
                    and (mean_elbow_angle <= eff_max_elbow)
                    and (shoulder_y <= bar_y + eff_shoulder_drop)
                )

                if state == "WAIT_HANG":
                    if hands_on_bar:
                        pre_pull_knee_err.append(1 if knees_bent else 0)
                        pre_pull_cross_err.append(1 if legs_crossed else 0)

                    if mean_elbow_angle >= elbow_ext_angle and hands_on_bar:
                        if hang_start_time is None:
                            hang_start_time = current_time
                        last_hang_duration = current_time - hang_start_time
                        hang_shoulder_y_ref = 0.85 * hang_shoulder_y_ref + 0.15 * shoulder_y
                        pre_bend_frames = 0
                        last_clean_hang_frame = clean_frame
                    else:
                        if (
                            CONFIG["pull_start_angle"] <= mean_elbow_angle < elbow_ext_angle
                            and hands_on_bar
                        ):
                            pre_bend_frames += 1
                            if hang_start_time is not None:
                                last_hang_duration = current_time - hang_start_time

                        shoulder_rising = (shoulder_y <= hang_shoulder_y_ref - 0.010)
                        if mean_elbow_angle < CONFIG["pull_start_angle"] and hands_on_bar and shoulder_rising:
                            state = "PULLING"
                            rep_start_time = current_time
                            rep_saved_hang_time = hang_start_time
                            rep_start_shoulder_y = hang_shoulder_y_ref
                            bar_is_locked = True

                            is_first_rep_early = (len(reps_log) == 0 and rep_start_time < 2.2)
                            rep_hang_pause_sec = max(1.0, last_hang_duration) if is_first_rep_early else last_hang_duration

                            min_elbow_in_rep = mean_elbow_angle
                            min_shoulder_y_in_rep = shoulder_y
                            min_chin_y_in_rep = chin_y
                            min_knee_in_rep = mean_knee_angle
                            min_hip_in_rep = mean_hip_angle
                            max_el_diff_in_rep = net_el_diff
                            curr_feet_ratio = ankle_dist / max(0.06, base_shoulder_w)
                            max_feet_apart_ratio = curr_feet_ratio
                            curr_ear_drop = ear_y - nose_y
                            max_ear_drop_in_rep = curr_ear_drop
                            max_elbow_in_descent = mean_elbow_angle
                            max_shoulder_y_in_descent = shoulder_y
                            descent_bottom_Start_time = None

                            chin_above_frames = 1 if chin_above_now else 0
                            head_thrown_back_frames = 1 if is_head_thrown_back else 0
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

                            if sum(pre_pull_knee_err) >= min_knee_f + 1:
                                rep_errors.add(f"нарушение ИП: согнутые в коленях ноги перед стартом (угол: {int(min_knee_in_rep)}° при норме ≥ {int(CONFIG['knee_bent_threshold'])}°)")
                            if sum(pre_pull_cross_err) >= min_cross_f + 1:
                                rep_errors.add("нарушение ИП: перекрещенные ноги перед стартом")

                            if not is_first_rep_early and rep_hang_pause_sec < min_hang_sec:
                                diff_hang = round(min_hang_sec - rep_hang_pause_sec, 2)
                                rep_errors.add(f"фиксация ИП менее {min_hang_sec:.2f} с ({rep_hang_pause_sec:.2f} с, не хватило: {max(0.01, diff_hang):.2f} с)")

                            hang_start_time = None
                            pre_bend_frames = 0

                elif state in ("PULLING", "DESCENDING"):
                    if not hands_on_bar:
                        rep_duration = current_time - rep_start_time
                        shoulder_rise = rep_start_shoulder_y - min_shoulder_y_in_rep
                        if (
                            rep_duration >= CONFIG["min_rep_duration"]
                            and min_elbow_in_rep <= CONFIG["min_pull_depth_angle"]
                            and shoulder_rise >= CONFIG["min_shoulder_rise_for_rep"]
                        ):
                            is_near_video_end = (total_frames > 0 and (total_frames - frame_idx) < int(0.8 * fps))
                            if not is_near_video_end:
                                rep_errors.add("срыв / спрыгивание с перекладины без фиксации виса")
                            _finalize_and_record_rep(current_time)
                            hang_start_time = None
                            last_hang_duration = 0.0
                        else:
                            hang_start_time = rep_saved_hang_time
                            if hang_start_time is not None:
                                last_hang_duration = current_time - hang_start_time
                        state = "WAIT_HANG"
                        pre_bend_frames = 0
                        rep_errors = set()
                        pre_pull_knee_err.clear()
                        pre_pull_cross_err.clear()
                    else:
                        rep_frames += 1

                        if state == "PULLING" and 75.0 <= mean_elbow_angle <= 110.0:
                            rep_mid_frame = clean_frame
                        if chin_y <= min_chin_y_in_rep or mean_elbow_angle <= min_elbow_in_rep:
                            rep_peak_frame = clean_frame

                        min_elbow_in_rep = min(min_elbow_in_rep, mean_elbow_angle)
                        min_shoulder_y_in_rep = min(min_shoulder_y_in_rep, shoulder_y)
                        min_chin_y_in_rep = min(min_chin_y_in_rep, chin_y)
                        min_knee_in_rep = min(min_knee_in_rep, mean_knee_angle)
                        min_hip_in_rep = min(min_hip_in_rep, mean_hip_angle)
                        max_el_diff_in_rep = max(max_el_diff_in_rep, net_el_diff)
                        curr_feet_ratio = ankle_dist / max(0.06, base_shoulder_w)
                        max_feet_apart_ratio = max(max_feet_apart_ratio, curr_feet_ratio)
                        curr_ear_drop = ear_y - nose_y
                        max_ear_drop_in_rep = max(max_ear_drop_in_rep, curr_ear_drop)
                        rep_ankle_x.append(ankle_center_x)

                        if chin_above_now:
                            chin_above_frames += 1
                        if is_head_thrown_back:
                            head_thrown_back_frames += 1

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
                            max_elbow_in_descent = mean_elbow_angle
                            max_shoulder_y_in_descent = shoulder_y
                            descent_bottom_Start_time = current_time

                        if state == "DESCENDING":
                            if mean_elbow_angle >= max_elbow_in_descent or shoulder_y >= max_shoulder_y_in_descent:
                                max_elbow_in_descent = max(max_elbow_in_descent, mean_elbow_angle)
                                max_shoulder_y_in_descent = max(max_shoulder_y_in_descent, shoulder_y)
                                descent_bottom_Start_time = current_time

                            full_hang_reached = mean_elbow_angle >= (elbow_ext_angle - 6.0)
                            total_rise = max(0.02, rep_start_shoulder_y - min_shoulder_y_in_rep)
                            descent_drop = max_shoulder_y_in_descent - min_shoulder_y_in_rep
                            rebound_pull_detected = (
                                (descent_drop >= 0.60 * total_rise or max_elbow_in_descent >= 124.0)
                                and (
                                    mean_elbow_angle < max_elbow_in_descent - 9.0
                                    or shoulder_y < max_shoulder_y_in_descent - 0.015
                                )
                            )
                            is_near_video_end = (total_frames > 0 and (total_frames - frame_idx) <= frame_step + 1)

                            if full_hang_reached or rebound_pull_detected or is_near_video_end:
                                rep_duration = current_time - rep_start_time
                                shoulder_rise = rep_start_shoulder_y - min_shoulder_y_in_rep

                                if (
                                    rep_duration >= CONFIG["min_rep_duration"]
                                    and min_elbow_in_rep <= CONFIG["min_pull_depth_angle"]
                                    and shoulder_rise >= CONFIG["min_shoulder_rise_for_rep"]
                                ):
                                    if rebound_pull_detected and not is_near_video_end and max_elbow_in_descent < (elbow_ext_angle - 10.0):
                                        rep_errors.add("неполное разгибание рук в нижней точке")

                                    _finalize_and_record_rep(current_time)

                                    if rebound_pull_detected and not full_hang_reached and not is_near_video_end:
                                        bounce_pause = max(0.10, current_time - (descent_bottom_Start_time or current_time))
                                        state = "PULLING"
                                        rep_start_time = current_time
                                        rep_saved_hang_time = current_time
                                        rep_hang_pause_sec = bounce_pause
                                        rep_start_shoulder_y = max(hang_shoulder_y_ref, max_shoulder_y_in_descent)

                                        min_elbow_in_rep = mean_elbow_angle
                                        min_shoulder_y_in_rep = shoulder_y
                                        min_chin_y_in_rep = chin_y
                                        min_knee_in_rep = mean_knee_angle
                                        min_hip_in_rep = mean_hip_angle
                                        max_el_diff_in_rep = net_el_diff
                                        curr_feet_ratio = ankle_dist / max(0.06, base_shoulder_w)
                                        max_feet_apart_ratio = curr_feet_ratio
                                        curr_ear_drop = ear_y - nose_y
                                        max_ear_drop_in_rep = curr_ear_drop
                                        max_elbow_in_descent = mean_elbow_angle
                                        max_shoulder_y_in_descent = shoulder_y
                                        chin_above_frames = 1 if chin_above_now else 0
                                        head_thrown_back_frames = 1 if is_head_thrown_back else 0
                                        diff_pause = round(min_hang_sec - bounce_pause, 2)
                                        rep_errors = {f"фиксация ИП менее {min_hang_sec:.2f} с ({bounce_pause:.2f} с, не хватило: {max(0.01, diff_pause):.2f} с)"}
                                        rep_frames = 1
                                        asym_frames = 1 if is_asym_frame else 0
                                        knee_bent_frames = 1 if knees_bent else 0
                                        legs_crossed_frames = 1 if legs_crossed else 0
                                        feet_apart_frames = 1 if feet_apart else 0
                                        kipping_frames = 1 if kipping_detected else 0
                                        rep_ankle_x = [ankle_center_x]
                                        rep_start_frame = clean_frame
                                        rep_mid_frame = clean_frame
                                        rep_peak_frame = clean_frame
                                    else:
                                        state = "WAIT_HANG"
                                        hang_start_time = current_time
                                        last_hang_duration = 0.0
                                        pre_bend_frames = 0
                                        rep_errors = set()
                                        pre_pull_knee_err.clear()
                                        pre_pull_cross_err.clear()
                                else:
                                    state = "WAIT_HANG"
                                    hang_start_time = rep_saved_hang_time if rep_saved_hang_time is not None else current_time
                                    last_hang_duration = current_time - hang_start_time
                                    pre_bend_frames = 0
                                    rep_errors = set()
                                    pre_pull_knee_err.clear()
                                    pre_pull_cross_err.clear()

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
                        or head_thrown_back_frames >= CONFIG.get("head_thrown_back_min_frames", 4)
                    )
                )
                ai_dispute_live = (
                    state != "WAIT_HANG"
                    and (
                        asym_frames >= min_asym_f
                        or legs_crossed_frames >= min_cross_f
                        or knee_bent_frames >= min_knee_f
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
        f"Повторение #{r['rep_number']} [{r['timestamp']}] | Подбородок выше грифа: {r.get('chin_above_bar', '✅ Да')} | "
        f"Пауза в висе: {r.get('hang_pause_sec', 0.0):.2f} с | "
        f"Вердикт судьи (ГТО / ВсОШ): {r['status_text']} ({', '.join(r['errors'])})"
        for r in reps_log
    ]

    result["total_attempts"] = len(reps_log)
    result["valid_reps"] = valid_count
    result["bar_y"] = round(bar_y if bar_y is not None else 0.155, 3)
    result["errors"] = all_unique_errors
    result["reps_detail"] = reps_log
    result["comment"] = "\n".join(summary_lines) if summary_lines else "Повторения не зафиксированы."
    result["tracked_video_path"] = final_video_path
    result["ai_referee_calls"] = ai_calls_count
    result["ai_referee_triggered"] = ai_calls_count > 0
    result["ai_verified_reps"] = ai_verified_list
    return result
