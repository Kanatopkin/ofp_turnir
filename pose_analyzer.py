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
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
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
    """Скачивает официальную модель MediaPipe Pose Landmarker при первом запуске."""
    model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
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


def draw_hud(frame, valid_reps, total_attempts, state, hang_sec, elbow_deg, knee_deg, bar_y, chin_y, active_errors):
    """Отрисовывает линию перекладины сверху и панель телеметрии (HUD) в левом нижнем углу."""
    h, w, _ = frame.shape

    # Линия перекладины (зеленая, когда подбородок взял высоту, иначе желтая)
    if bar_y is not None:
        bar_px = int(bar_y * h)
        chin_above = (chin_y is not None) and (chin_y <= bar_y + 0.025)
        bar_color = (0, 255, 0) if chin_above else (0, 215, 255)
        cv2.line(frame, (0, bar_px), (w, bar_px), bar_color, 3)
        label_y = bar_px + 22 if bar_px < 35 else bar_px - 8
        cv2.putText(
            frame,
            f"BAR LINE (Y={bar_y:.2f}) {'[CHIN ABOVE]' if chin_above else ''}",
            (15, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2
        )

    # Полупрозрачная плашка телеметрии в ЛЕВОМ НИЖНЕМ углу, чтобы не закрывать турник и руки
    overlay = frame.copy()
    box_w = min(w - 20, 430)
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
            lw_y = lm[LM.LEFT_WRIST].y
            rw_y = lm[LM.RIGHT_WRIST].y
            detected_bar_y = float((lw_y + rw_y) / 2.0)
            nose = [lm[LM.NOSE].x, lm[LM.NOSE].y]
            mouth_x = (lm[LM.MOUTH_LEFT].x + lm[LM.MOUTH_RIGHT].x) / 2.0
            mouth_y = (lm[LM.MOUTH_LEFT].y + lm[LM.MOUTH_RIGHT].y) / 2.0
            chin_y = mouth_y + 0.22 * abs(mouth_y - nose[1])
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
    min_hang_sec=0.85,
    generate_tracked_video=True,
    progress_callback=None
):
    """
    Покадровый анализ видео на основе MediaPipe PoseLandmarker и конечного автомата (State Machine)
    с защитой от перспективных искажений ракурса камеры и фильтрацией спрыгивания.
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
    bar_y = manual_bar_y

    rep_start_time = 0.0
    min_chin_y = 1.0
    min_elbow_in_rep = 180.0
    rep_errors = set()

    # Счетчики кадров внутри одного повторения (чтобы единичный шум кадра не браковал попытку)
    rep_frames = 0
    asym_frames = 0
    knee_bent_frames = 0
    legs_crossed_frames = 0
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
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
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
            mean_knee_angle = 0.0
            chin_y = None

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
                chin_y = mouth_y + 0.22 * abs(mouth_y - nose[1])

                wrist_y = (l_wr[1] + r_wr[1]) / 2.0
                wrist_x = (l_wr[0] + r_wr[0]) / 2.0
                if manual_bar_y is None and len(bar_y_samples) < 60:
                    bar_y_samples.append(wrist_y)
                    bar_y = float(np.median(bar_y_samples))

                angle_el_l = calculate_angle(l_sh, l_el, l_wr)
                angle_el_r = calculate_angle(r_sh, r_el, r_wr)
                mean_elbow_angle = (angle_el_l + angle_el_r) / 2.0

                angle_kn_l = calculate_angle(l_hip, l_kn, l_an)
                angle_kn_r = calculate_angle(r_hip, r_kn, r_an)
                mean_knee_angle = (angle_kn_l + angle_kn_r) / 2.0

                angle_hip_l = calculate_angle(l_sh, l_hip, l_kn)
                angle_hip_r = calculate_angle(r_sh, r_hip, r_kn)
                mean_hip_angle = (angle_hip_l + angle_hip_r) / 2.0

                shoulder_width = max(0.06, abs(l_sh[0] - r_sh[0]))

                # Устойчивые критерии нарушений с защитой от бокового ракурса камеры:
                # 1. Явный перекос плеч и локтей по высоте + разница углов в средней фазе тяги
                shoulder_tilt = abs(l_sh[1] - r_sh[1]) / shoulder_width
                elbow_diff = abs(angle_el_l - angle_el_r)
                is_asym_frame = (
                    (60.0 < mean_elbow_angle < 145.0)
                    and (elbow_diff > 38.0)
                    and (shoulder_tilt > 0.35)
                )

                # 2. Сгибание коленей и скрещивание/разведение стоп
                ankle_dist = abs(l_an[0] - r_an[0])
                legs_crossed = (l_an[0] - r_an[0]) * (l_hip[0] - r_hip[0]) < -0.015
                feet_apart = ankle_dist > (shoulder_width * 1.15)
                knees_bent = mean_knee_angle < 150.0

                # 3. Рывки и махи (сильное сгибание в тазу или раскачка корпуса)
                hip_center_x = (l_hip[0] + r_hip[0]) / 2.0
                horizontal_swing = abs(hip_center_x - wrist_x) > (shoulder_width * 0.95)
                kipping_detected = (mean_hip_angle < 142.0) or horizontal_swing

                # Проверка, что руки находятся на перекладине (участник не спрыгнул на землю)
                hands_on_bar = abs(wrist_y - bar_y) < 0.14

                if state == "WAIT_HANG":
                    if mean_elbow_angle >= elbow_ext_angle and hands_on_bar:
                        if hang_start_time is None:
                            hang_start_time = current_time
                        last_hang_duration = current_time - hang_start_time
                    else:
                        if mean_elbow_angle < 150.0 and chin_y > bar_y and hands_on_bar:
                            state = "PULLING"
                            rep_start_time = current_time
                            min_chin_y = chin_y
                            min_elbow_in_rep = mean_elbow_angle
                            rep_errors = set()
                            rep_frames = 0
                            asym_frames = 0
                            knee_bent_frames = 0
                            legs_crossed_frames = 0
                            kipping_frames = 0

                            # Если это самое первое повторение на первых 2 секундах видео,
                            # не штрафуем за позднее включение камеры оператором
                            is_first_rep_early = (len(reps_log) == 0 and rep_start_time < 2.0)
                            if not is_first_rep_early and last_hang_duration < min_hang_sec:
                                rep_errors.add(f"фиксация ИП менее 1.0 сек ({last_hang_duration:.2f} с)")
                            hang_start_time = None

                elif state in ("PULLING", "DESCENDING"):
                    # Если участник отпустил перекладину и спрыгнул вниз — отменяем ложную попытку
                    if not hands_on_bar:
                        state = "WAIT_HANG"
                        hang_start_time = None
                        last_hang_duration = 0.0
                    else:
                        rep_frames += 1
                        min_chin_y = min(min_chin_y, chin_y)
                        min_elbow_in_rep = min(min_elbow_in_rep, mean_elbow_angle)

                        if is_asym_frame:
                            asym_frames += 1
                        if knees_bent:
                            knee_bent_frames += 1
                        if legs_crossed or feet_apart:
                            legs_crossed_frames += 1
                        if kipping_detected:
                            kipping_frames += 1

                        if state == "PULLING" and (chin_y > min_chin_y + 0.03 or mean_elbow_angle < 105.0):
                            state = "DESCENDING"

                        if state == "DESCENDING" and mean_elbow_angle >= (elbow_ext_angle - 4.0):
                            rep_duration = current_time - rep_start_time

                            # Фильтруем случайные микродвижения короче 0.55 сек без реального подъема
                            if rep_duration >= 0.55 and min_elbow_in_rep < 135.0:
                                min_f = max(4, int(rep_frames * 0.30))
                                if asym_frames >= min_f:
                                    rep_errors.add("поочередное (неравномерное) сгибание рук")
                                if knee_bent_frames >= min_f:
                                    rep_errors.add("нарушение ИП: согнутые в коленях ноги")
                                if legs_crossed_frames >= min_f:
                                    rep_errors.add("нарушение ИП: перекрещенные или разведенные ноги")
                                if kipping_frames >= max(3, int(rep_frames * 0.25)):
                                    rep_errors.add("подтягивание рывками или с махом ногами/туловищем")

                                # Допуск 0.025 (2.5% высоты кадра) учитывает толщину кистей на грифе
                                if min_chin_y > (bar_y + 0.025):
                                    rep_errors.add("подбородок ниже уровня грифа перекладины")

                                is_valid = len(rep_errors) == 0
                                reps_log.append({
                                    "rep_number": len(reps_log) + 1,
                                    "timestamp": f"{format_timestamp(rep_start_time)} – {format_timestamp(current_time)}",
                                    "valid": is_valid,
                                    "status_text": "✅ Засчитано" if is_valid else "❌ Не засчитано",
                                    "errors": list(rep_errors) if rep_errors else ["Нет ошибок"]
                                })

                            state = "WAIT_HANG"
                            hang_start_time = current_time
                            last_hang_duration = 0.0

                if writer is not None:
                    draw_skeleton(frame, lm, chin_pt=(mouth_x, chin_y))

            if writer is not None:
                valid_so_far = sum(1 for r in reps_log if r["valid"])
                draw_hud(
                    frame,
                    valid_reps=valid_so_far,
                    total_attempts=len(reps_log),
                    state=state,
                    hang_sec=last_hang_duration,
                    elbow_deg=mean_elbow_angle,
                    knee_deg=mean_knee_angle,
                    bar_y=bar_y,
                    chin_y=chin_y,
                    active_errors=bool(rep_errors) if state != "WAIT_HANG" else False
                )
                writer.write(frame)

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

    return {
        "total_attempts": len(reps_log),
        "valid_reps": valid_count,
        "bar_y": round(bar_y if bar_y is not None else 0.25, 3),
        "errors": all_unique_errors,
        "reps_detail": reps_log,
        "comment": "\n".join(summary_lines) if summary_lines else "Повторения не зафиксированы.",
        "tracked_video_path": final_video_path
    }
