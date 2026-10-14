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
    model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


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
    return float(min(knuckle_y, wrist_y - 0.018))


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
            chin_y = mouth_y + 0.45 * abs(mouth_y - nose[1])
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
    elbow_ext_angle=162.0,
    min_hang_sec=0.90,
    generate_tracked_video=True,
    progress_callback=None
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "total_attempts": 0,
            "valid_reps": 0,
            "bar_y": 0.25,
            "errors": [],
            "reps_detail": [],
            "comment": "Не удалось открыть видеофайл.",
            "tracked_video_path": None
        }

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

    state = "WAIT_HANG"
    hang_start_time = None
    last_hang_duration = 0.0

    bar_y_samples = []
    shoulder_w_samples = []
    bar_y = manual_bar_y if manual_bar_y is not None else 0.12
    base_shoulder_w = 0.12

    chin_buf = deque(maxlen=3)
    elbow_buf = deque(maxlen=3)
    knee_buf = deque(maxlen=3)
    shoulder_y_buf = deque(maxlen=3)

    pre_pull_knee_err = deque(maxlen=25)
    pre_pull_cross_err = deque(maxlen=25)

    rep_start_time = 0.0
    min_elbow_in_rep = 180.0
    min_shoulder_y_in_rep = 1.0
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

                angle_kn_2d = (calculate_angle(l_hip, l_kn, l_an) + calculate_angle(r_hip, r_kn, r_an)) / 2.0
                thigh_len = (np.hypot(l_kn[0] - l_hip[0], l_kn[1] - l_hip[1]) + np.hypot(r_kn[0] - r_hip[0], r_kn[1] - r_hip[1])) / 2.0
                shin_len = (np.hypot(l_an[0] - l_kn[0], l_an[1] - l_kn[1]) + np.hypot(r_an[0] - r_kn[0], r_an[1] - r_kn[1])) / 2.0
                shin_ratio = shin_len / max(0.04, thigh_len)

                if shin_ratio < 0.68:
                    sagittal_knee_deg = float(np.degrees(np.arcsin(np.clip(shin_ratio / 0.85, 0.1, 1.0))) * 1.6)
                    raw_knee = min(angle_kn_2d, sagittal_knee_deg)
                else:
                    raw_knee = angle_kn_2d

                chin_buf.append(raw_chin_y)
                elbow_buf.append(raw_elbow)
                knee_buf.append(raw_knee)
                shoulder_y_buf.append(raw_shoulder_y)

                chin_y = float(np.median(chin_buf))
                mean_elbow_angle = float(np.median(elbow_buf))
                mean_knee_angle = float(np.median(knee_buf))
                shoulder_y = float(np.median(shoulder_y_buf))

                mean_hip_angle = (calculate_angle(l_sh, l_hip, l_kn) + calculate_angle(r_sh, r_hip, r_kn)) / 2.0

                curr_sh_w = abs(l_sh[0] - r_sh[0])
                if mean_elbow_angle > 160.0 and curr_sh_w > 0.05 and len(shoulder_w_samples) < 60:
                    shoulder_w_samples.append(curr_sh_w)
                    base_shoulder_w = float(np.median(shoulder_w_samples))

                shoulder_tilt = abs(l_sh[1] - r_sh[1]) / max(0.06, base_shoulder_w)
                elbow_tilt = abs(l_el[1] - r_el[1]) / max(0.06, base_shoulder_w)
                elbow_diff = abs(angle_el_l - angle_el_r)
                is_asym_frame = (
                    (25.0 < mean_elbow_angle < 150.0)
                    and (elbow_diff > 19.0 or elbow_tilt > 0.22 or shoulder_tilt > 0.18)
                )

                ankle_dx = l_an[0] - r_an[0]
                knee_dx = l_kn[0] - r_kn[0]
                hip_dx = l_hip[0] - r_hip[0]
                ankle_dist = abs(ankle_dx)
                ankle_dy = abs(l_an[1] - r_an[1])

                shins_intersect = (ankle_dx * knee_dx < 0.0001) or (ankle_dx * hip_dx < 0.0001)
                ankles_stacked = (ankle_dist < base_shoulder_w * 0.22) and (ankle_dy > 0.016)
                legs_crossed = shins_intersect or ankles_stacked

                feet_apart = ankle_dist > (base_shoulder_w * 0.95)
                knees_bent = (mean_knee_angle < 158.0) or (shin_ratio < 0.68)

                hip_center_x = (l_hip[0] + r_hip[0]) / 2.0
                ankle_center_x = (l_an[0] + r_an[0]) / 2.0
                horizontal_swing = (
                    abs(hip_center_x - wrist_x) > (base_shoulder_w * 0.65)
                    or abs(ankle_center_x - hip_center_x) > (base_shoulder_w * 0.55)
                )
                kipping_detected = (mean_hip_angle < 160.0) or horizontal_swing
                hands_on_bar = abs(wrist_y - bar_y) < 0.16

                chin_above_now = (
                    (chin_y <= bar_y - 0.008)
                    and (mean_elbow_angle <= 45.0)
                    and (shoulder_y <= bar_y + 0.11)
                )

                if frame_idx <= 5 and state == "WAIT_HANG" and mean_elbow_angle < 60.0 and hands_on_bar:
                    state = "DESCENDING"
                    rep_start_time = 0.0
                    min_elbow_in_rep = mean_elbow_angle
                    min_shoulder_y_in_rep = shoulder_y
                    chin_above_frames = 3 if chin_above_now else 0
                    rep_errors = set()
                    rep_frames = 1
                    asym_frames = knee_bent_frames = legs_crossed_frames = feet_apart_frames = kipping_frames = 0

                elif state == "WAIT_HANG":
                    pre_pull_knee_err.append(1 if knees_bent else 0)
                    pre_pull_cross_err.append(1 if legs_crossed else 0)

                    if mean_elbow_angle >= elbow_ext_angle and hands_on_bar:
                        if hang_start_time is None:
                            hang_start_time = current_time
                        last_hang_duration = current_time - hang_start_time
                    else:
                        if mean_elbow_angle < 150.0 and hands_on_bar:
                            state = "PULLING"
                            rep_start_time = current_time
                            min_elbow_in_rep = mean_elbow_angle
                            min_shoulder_y_in_rep = shoulder_y
                            chin_above_frames = 1 if chin_above_now else 0
                            rep_errors = set()
                            rep_frames = 1
                            asym_frames = knee_bent_frames = legs_crossed_frames = feet_apart_frames = kipping_frames = 0

                            if sum(pre_pull_knee_err) >= 5:
                                rep_errors.add("нарушение ИП: согнутые в коленях ноги")
                            if sum(pre_pull_cross_err) >= 6:
                                rep_errors.add("нарушение ИП: перекрещенные ноги")

                            is_first_rep_early = (len(reps_log) == 0 and rep_start_time < 2.0)
                            if not is_first_rep_early and last_hang_duration < min_hang_sec:
                                rep_errors.add(f"фиксация ИП менее 1.0 сек ({last_hang_duration:.2f} с)")
                            hang_start_time = None

                elif state in ("PULLING", "DESCENDING"):
                    if not hands_on_bar:
                        state = "WAIT_HANG"
                        hang_start_time = None
                        last_hang_duration = 0.0
                    else:
                        rep_frames += 1
                        min_elbow_in_rep = min(min_elbow_in_rep, mean_elbow_angle)
                        min_shoulder_y_in_rep = min(min_shoulder_y_in_rep, shoulder_y)
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
                            or shoulder_y > min_shoulder_y_in_rep + 0.03
                        ):
                            state = "DESCENDING"

                        if state == "DESCENDING" and mean_elbow_angle >= (elbow_ext_angle - 3.0):
                            rep_duration = current_time - rep_start_time
                            if rep_duration >= 0.45 and min_elbow_in_rep <= 135.0:
                                if asym_frames >= max(4, int(rep_frames * 0.22)):
                                    rep_errors.add("поочередное (неравномерное) сгибание рук")
                                if knee_bent_frames >= max(3, int(rep_frames * 0.18)):
                                    rep_errors.add("нарушение ИП: согнутые в коленях ноги")
                                if legs_crossed_frames >= max(4, int(rep_frames * 0.22)):
                                    rep_errors.add("нарушение ИП: перекрещенные ноги")
                                if feet_apart_frames >= max(5, int(rep_frames * 0.28)):
                                    rep_errors.add("нарушение ИП: ступни не вместе (ноги разведены)")
                                if kipping_frames >= max(4, int(rep_frames * 0.22)):
                                    rep_errors
