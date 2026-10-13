import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


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
    if not ret:
        cap.release()
        return None, 0.25

    h, w, _ = frame.shape
    detected_bar_y = 0.25

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            lw_y = lm[mp_pose.PoseLandmark.LEFT_WRIST].y
            rw_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
            detected_bar_y = float((lw_y + rw_y) / 2.0)
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cap.release()
    bar_y = manual_bar_y if manual_bar_y is not None else detected_bar_y
    bar_px = int(bar_y * h)

    cv2.line(frame, (0, bar_px), (w, bar_px), (0, 255, 0), 3)
    cv2.putText(
        frame, f"Bar Line Y={bar_y:.2f}", (20, max(30, bar_px - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round(detected_bar_y, 3)


def analyze_video_pose(
    video_path,
    manual_bar_y=None,
    elbow_ext_angle=170.0,
    min_hang_sec=1.0,
    max_asym_deg=22.0,
    progress_callback=None
):
    """
    Покадровый анализ видео на основе MediaPipe Pose и конечного автомата (State Machine).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    state = "WAIT_HANG"  # WAIT_HANG -> PULLING -> DESCENDING
    hang_start_time = None
    last_hang_duration = 0.0

    bar_y_samples = []
    bar_y = manual_bar_y

    rep_start_time = 0.0
    min_chin_y = 1.0
    rep_errors = set()
    prev_hip_angle = None
    prev_time = None

    reps_log = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = frame_idx / fps

            if progress_callback and total_frames > 0 and frame_idx % 10 == 0:
                progress_callback(min(1.0, frame_idx / total_frames))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark

            def pt(landmark_enum):
                p = lm[landmark_enum]
                return [p.x, p.y]

            l_sh, r_sh = pt(mp_pose.PoseLandmark.LEFT_SHOULDER), pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            l_el, r_el = pt(mp_pose.PoseLandmark.LEFT_ELBOW), pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
            l_wr, r_wr = pt(mp_pose.PoseLandmark.LEFT_WRIST), pt(mp_pose.PoseLandmark.RIGHT_WRIST)
            l_hip, r_hip = pt(mp_pose.PoseLandmark.LEFT_HIP), pt(mp_pose.PoseLandmark.RIGHT_HIP)
            l_kn, r_kn = pt(mp_pose.PoseLandmark.LEFT_KNEE), pt(mp_pose.PoseLandmark.RIGHT_KNEE)
            l_an, r_an = pt(mp_pose.PoseLandmark.LEFT_ANKLE), pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

            nose = pt(mp_pose.PoseLandmark.NOSE)
            m_l = pt(mp_pose.PoseLandmark.MOUTH_LEFT)
            m_r = pt(mp_pose.PoseLandmark.MOUTH_RIGHT)
            mouth_y = (m_l[1] + m_r[1]) / 2.0
            chin_y = mouth_y + 0.35 * abs(mouth_y - nose[1])

            wrist_y = (l_wr[1] + r_wr[1]) / 2.0
            wrist_x = (l_wr[0] + r_wr[0]) / 2.0
            if manual_bar_y is None:
                bar_y_samples.append(wrist_y)
                bar_y = float(np.median(bar_y_samples[:45]))

            angle_el_l = calculate_angle(l_sh, l_el, l_wr)
            angle_el_r = calculate_angle(r_sh, r_el, r_wr)
            mean_elbow_angle = (angle_el_l + angle_el_r) / 2.0
            elbow_diff = abs(angle_el_l - angle_el_r)

            angle_kn_l = calculate_angle(l_hip, l_kn, l_an)
            angle_kn_r = calculate_angle(r_hip, r_kn, r_an)
            mean_knee_angle = (angle_kn_l + angle_kn_r) / 2.0

            angle_hip_l = calculate_angle(l_sh, l_hip, l_kn)
            angle_hip_r = calculate_angle(r_sh, r_hip, r_kn)
            mean_hip_angle = (angle_hip_l + angle_hip_r) / 2.0

            shoulder_width = max(0.05, abs(l_sh[0] - r_sh[0]))
            ankle_dist = abs(l_an[0] - r_an[0])
            legs_crossed = (l_an[0] - r_an[0]) * (l_hip[0] - r_hip[0]) < -0.002
            feet_apart = ankle_dist > (shoulder_width * 0.85)
            knees_bent = mean_knee_angle < 162.0

            hip_center_x = (l_hip[0] + r_hip[0]) / 2.0
            horizontal_swing = abs(hip_center_x - wrist_x) > (shoulder_width * 0.75)

            hip_vel = 0.0
            if prev_hip_angle is not None and prev_time is not None:
                dt = current_time - prev_time
                if dt > 0:
                    hip_vel = abs(mean_hip_angle - prev_hip_angle) / dt
            prev_hip_angle = mean_hip_angle
            prev_time = current_time

            kipping_detected = (mean_hip_angle < 155.0) or (hip_vel > 160.0) or horizontal_swing

            if state == "WAIT_HANG":
                if mean_elbow_angle >= elbow_ext_angle:
                    if hang_start_time is None:
                        hang_start_time = current_time
                    last_hang_duration = current_time - hang_start_time
                else:
                    if mean_elbow_angle < 158.0 and chin_y > bar_y:
                        state = "PULLING"
                        rep_start_time = current_time
                        min_chin_y = chin_y
                        rep_errors = set()

                        if last_hang_duration < min_hang_sec:
                            rep_errors.add(f"фиксация ИП менее {min_hang_sec:.1f} сек ({last_hang_duration:.2f} с)")
                        hang_start_time = None

            elif state in ("PULLING", "DESCENDING"):
                min_chin_y = min(min_chin_y, chin_y)

                if elbow_diff > max_asym_deg:
                    rep_errors.add("поочередное (неравномерное) сгибание рук")
                if knees_bent:
                    rep_errors.add("нарушение ИП: согнутые в коленях ноги")
                if legs_crossed or feet_apart:
                    rep_errors.add("нарушение ИП: перекрещенные или разведенные ноги")
                if kipping_detected:
                    rep_errors.add("подтягивание рывками или с махом ногами/туловищем")

                if state == "PULLING" and (chin_y > min_chin_y + 0.03 or mean_elbow_angle < 105.0):
                    state = "DESCENDING"

                if state == "DESCENDING" and mean_elbow_angle >= (elbow_ext_angle - 3.0):
                    if min_chin_y >= bar_y:
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

    cap.release()

    if state == "DESCENDING":
        rep_errors.add("неполное разгибание рук в финальной фазе опускания")
        if min_chin_y >= bar_y:
            rep_errors.add("подбородок ниже уровня грифа перекладины")
        reps_log.append({
            "rep_number": len(reps_log) + 1,
            "timestamp": f"{format_timestamp(rep_start_time)} – конец видео",
            "valid": False,
            "status_text": "❌ Не засчитано",
            "errors": list(rep_errors)
        })

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
        "comment": "\n".join(summary_lines) if summary_lines else "Повторения не зафиксированы."
    }
