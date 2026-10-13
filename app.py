import streamlit as st
import pandas as pd
import json
import io
import os
import ssl
import tempfile
import time
from datetime import datetime
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

from pose_analyzer import analyze_video_pose, get_calibration_preview

st.set_page_config(
    page_title="Турнир по подтягиванию | Pose Estimation",
    page_icon="🏆",
    layout="wide"
)

DB_FILENAME = "tournament_db.json"


def get_drive_service():
    """Создает свежее подключение к Google Drive API для предотвращения SSL RECORD_LAYER_FAILURE."""
    if "gdrive_oauth" in st.secrets:
        creds = Credentials(
            token=None,
            refresh_token=st.secrets["gdrive_oauth"]["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=st.secrets["gdrive_oauth"]["client_id"],
            client_secret=st.secrets["gdrive_oauth"]["client_secret"],
            scopes=["https://www.googleapis.com/auth/drive"]
        )
    else:
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=["https://www.googleapis.com/auth/drive"]
        )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def retry_drive_op(func, retries=3, delay=1.0):
    """Выполняет операцию Google Drive с автоматическим переподключением при сбоях SSL/сети."""
    last_err = None
    for attempt in range(retries):
        try:
            return func()
        except (ssl.SSLError, OSError, Exception) as e:
            last_err = e
            time.sleep(delay * (attempt + 1))
    raise last_err


def load_db():
    def _op():
        service = get_drive_service()
        folder_id = st.secrets["general"]["drive_folder_id"]
        query = f"'{folder_id}' in parents and name = '{DB_FILENAME}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=3)
        files = results.get("files", [])
        if not files:
            return []
        request = service.files().get_media(fileId=files[0]["id"])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=3)
        fh.seek(0)
        return json.loads(fh.read().decode("utf-8"))

    return retry_drive_op(_op)


def save_db(data):
    def _op():
        service = get_drive_service()
        folder_id = st.secrets["general"]["drive_folder_id"]
        query = f"'{folder_id}' in parents and name = '{DB_FILENAME}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=3)
        files = results.get("files", [])
        json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        media = MediaIoBaseUpload(io.BytesIO(json_bytes), mimetype="application/json", resumable=True)
        if files:
            service.files().update(fileId=files[0]["id"], media_body=media).execute(num_retries=3)
        else:
            file_metadata = {"name": DB_FILENAME, "parents": [folder_id]}
            service.files().create(body=file_metadata, media_body=media, fields="id").execute(num_retries=3)

    return retry_drive_op(_op)


def upload_video_to_drive(uploaded_file, filename):
    def _op():
        service = get_drive_service()
        folder_id = st.secrets["general"]["drive_folder_id"]
        file_metadata = {"name": filename, "parents": [folder_id]}
        uploaded_file.seek(0)
        media = MediaIoBaseUpload(uploaded_file, mimetype=uploaded_file.type, resumable=True)
        file = service.files().create(
            body=file_metadata, media_body=media, fields="id, webViewLink"
        ).execute(num_retries=3)
        try:
            service.permissions().create(
                fileId=file["id"], body={"type": "anyone", "role": "reader"}
            ).execute(num_retries=2)
        except Exception:
            pass
        return file["id"], file["webViewLink"]

    return retry_drive_op(_op)


def download_video_temp(file_id):
    def _op():
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        downloader = MediaIoBaseDownload(tmp, request)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=3)
        tmp.close()
        return tmp.name

    return retry_drive_op(_op)


if "db" not in st.session_state:
    try:
        st.session_state.db = load_db()
    except Exception as e:
        st.session_state.db = []
        st.warning(f"Ожидание настройки подключения к Google Drive в Secrets: {e}")

if "tracked_videos" not in st.session_state:
    st.session_state.tracked_videos = {}

st.title("🏆 Турнир по подтягиванию на перекладине памяти героев СВО")
st.markdown(
    "**Дистанционный формат (20–27 октября)** | **Участники:** 8–9 классы (Московская область) | "
    "**Команда:** 10–12 человек (зачет по сумме 10 лучших)"
)

with st.expander("🤖 Архитектура компьютерного зрения (Pose Estimation) и критерии"):
    st.markdown("""
    * **Отслеживание ключевых точек (MediaPipe Pose):** кисти, локти, плечи, подбородок, тазобедренные суставы, колени и лодыжки.
    * **Контроль исходного положения (ИП):** разгибание локтей 170°–180°, прямые ноги (≈180°), сведенные стопы без скрещивания, обязательная фиксация неподвижного виса **не менее 1,0 секунды** перед началом каждого подъема.
    * **Контроль подъема и верхней точки:** пересечение координатой Y подбородка горизонтальной линии грифа перекладины, симметрия сгибания левой и правой руки, отсутствие рывков и махов тазом/ногами.
    """)

st.divider()

col_reg, col_board = st.columns([1, 1.4], gap="large")

with col_reg:
    st.subheader("📥 Регистрация и загрузка видео")
    with st.form("registration_form", clear_on_submit=True):
        school = st.text_input("Школа (Московская область)*", placeholder="МБОУ СОШ №1").strip()
        fio = st.text_input("ФИО участника*", placeholder="Иванов Иван Иванович").strip()
        grade = st.selectbox("Класс*", ["8 класс", "9 класс"])
        self_reps = st.number_input("Заявленное количество подтягиваний", min_value=0, max_value=100, value=10)
        video_file = st.file_uploader("Видеозапись выполнения (MP4, MOV)*", type=["mp4", "mov", "avi", "mkv"])
        submitted = st.form_submit_button("Загрузить видео в Google Drive", use_container_width=True)

        if submitted:
            if not school or not fio or not video_file:
                st.error("Заполните все обязательные поля и прикрепите видео.")
            else:
                school_entries = [x for x in st.session_state.db if x["school"].lower() == school.lower()]
                if len(school_entries) >= 12:
                    st.error(f"Лимит исчерпан! От школы «{school}» уже загружено 12 участников.")
                else:
                    with st.spinner("Сохранение видео в Google Drive..."):
                        safe_name = f"{school}_{grade}_{fio}_{datetime.now().strftime('%m%d_%H%M')}.mp4"
                        file_id, web_link = upload_video_to_drive(video_file, safe_name)
                        new_entry = {
                            "id": f"id_{int(time.time()*1000)}",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "school": school,
                            "fio": fio,
                            "grade": grade,
                            "claimed_reps": self_reps,
                            "verified_reps": None,
                            "status": "На проверке",
                            "bar_y": None,
                            "errors": [],
                            "reps_detail": [],
                            "comment": "",
                            "file_id": file_id,
                            "video_url": web_link
                        }
                        st.session_state.db.append(new_entry)
                        save_db(st.session_state.db)
                        st.success(f"Участник {fio} зарегистрирован! Видео загружено в Google Drive.")

with col_board:
    col_title, col_sync = st.columns([3, 1])
    with col_title:
        st.subheader("📊 Онлайн-табло результатов")
    with col_sync:
        if st.button("🔄 Обновить", use_container_width=True):
            st.session_state.db = load_db()
            st.rerun()

    if not st.session_state.db:
        st.info("Заявки пока не поступали.")
    else:
        df = pd.DataFrame(st.session_state.db)
        df["score"] = df["verified_reps"].fillna(0).astype(int)
        tab_team, tab_ind = st.tabs(["🏫 Командный зачет (Сумма 10 лучших)", "🥇 Личный зачет"])

        with tab_team:
            team_rows = []
            for sch, group in df.groupby("school"):
                total_m = len(group)
                checked_m = len(group[group["status"] == "Проверено"])
                top10_sum = group.sort_values(by="score", ascending=False).head(10)["score"].sum()
                team_rows.append({
                    "Школа": sch,
                    "Участников (квота 10–12)": f"{total_m} чел.",
                    "Проверено": f"{checked_m} из {total_m}",
                    "Зачет команды (≥10 чел.)": "✅ Допущена" if total_m >= 10 else "⏳ Некомплект",
                    "Сумма 10 лучших": top10_sum
                })
            st.dataframe(
                pd.DataFrame(team_rows).sort_values(by="Сумма 10 лучших", ascending=False),
                use_container_width=True, hide_index=True
            )

        with tab_ind:
            df_ind = df[["fio", "school", "grade", "claimed_reps", "verified_reps", "status"]].copy()
            df_ind.columns = ["ФИО", "Школа", "Класс", "Заявлено", "Засчитано", "Статус"]
            st.dataframe(
                df_ind.sort_values(by="Засчитано", ascending=False, na_position="last"),
                use_container_width=True, hide_index=True
            )

st.divider()

st.subheader("⚖️ Судейский модуль (Pose Estimation & State Machine)")
judge_pwd = st.text_input("Введите пароль судьи для калибровки, запуска трекинга и утверждения протоколов:", type="password")

if judge_pwd:
    if judge_pwd == st.secrets["general"]["judge_password"]:
        st.success("Авторизация судьи выполнена успешно.")

        if st.session_state.db:
            c_filt, c_exp = st.columns([2, 1])
            with c_filt:
                filter_status = st.radio("Фильтр списка:", ["Все", "На проверке", "Проверено"], horizontal=True)
            with c_exp:
                df_exp = pd.DataFrame(st.session_state.db).drop(columns=["reps_detail"], errors="ignore")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_exp.to_excel(writer, index=False, sheet_name="Протокол")
                st.download_button(
                    "📥 Скачать сводный протокол (Excel)",
                    data=buf.getvalue(),
                    file_name=f"Протокол_Подтягивание_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            filtered_list = [
                x for x in st.session_state.db
                if filter_status == "Все" or x["status"] == filter_status
            ]

            for entry in filtered_list:
                with st.expander(
                    f"{'🟢' if entry['status']=='Проверено' else '🟡'} {entry['school']} — {entry['fio']} ({entry['grade']}) | "
                    f"Заявлено: {entry['claimed_reps']} | Засчитано: {entry['verified_reps'] if entry['verified_reps'] is not None else '—'}"
                ):
                    col_left, col_right = st.columns([1.2, 1.3])

                    with col_left:
                        st.markdown(f"🔗 **[Открыть исходное видео в Google Drive]({entry['video_url']})**")
                        use_manual_bar = st.checkbox(
                            "Задать линию перекладины вручную (калибровка)",
                            value=(entry.get("bar_y") is not None),
                            key=f"chk_bar_{entry['id']}"
                        )
                        manual_bar_val = None
                        if use_manual_bar:
                            manual_bar_val = st.slider(
                                "Высота грифа перекладины Y (0.0 — верх кадра, 1.0 — низ кадра):",
                                min_value=0.05, max_value=0.80,
                                value=float(entry["bar_y"]) if entry.get("bar_y") else 0.25,
                                step=0.01,
                                key=f"slider_bar_{entry['id']}"
                            )

                        b_prev, b_run = st.columns(2)
                        with b_prev:
                            if st.button("👁️ Калибровка кадра", key=f"prev_{entry['id']}", use_container_width=True):
                                with st.spinner("Загрузка кадра и поиск кистей..."):
                                    tmp_path = download_video_temp(entry["file_id"])
                                    try:
                                        img, auto_y = get_calibration_preview(tmp_path, manual_bar_val)
                                        if img is not None:
                                            st.image(img, caption=f"Линия грифа (авто-определение по кистям: Y={auto_y})", use_container_width=True)
                                    finally:
                                        if os.path.exists(tmp_path):
                                            os.remove(tmp_path)

                        with b_run:
                            if st.button("🤖 Анализ + Видео с трекингом", key=f"pose_{entry['id']}", use_container_width=True):
                                prog_bar = st.progress(0.0, text="Покадровая обработка и запись видео со скелетом...")
                                tmp_path = download_video_temp(entry["file_id"])
                                try:
                                    res = analyze_video_pose(
                                        tmp_path,
                                        manual_bar_y=manual_bar_val,
                                        generate_tracked_video=True,
                                        progress_callback=lambda p: prog_bar.progress(p, text=f"Трекинг ключевых точек: {int(p*100)}%")
                                    )
                                    prog_bar.empty()
                                    entry["verified_reps"] = res["valid_reps"]
                                    entry["bar_y"] = res["bar_y"]
                                    entry["errors"] = res["errors"]
                                    entry["reps_detail"] = res["reps_detail"]
                                    entry["comment"] = res["comment"]

                                    if res.get("tracked_video_path") and os.path.exists(res["tracked_video_path"]):
                                        with open(res["tracked_video_path"], "rb") as vf:
                                            st.session_state.tracked_videos[entry["id"]] = vf.read()
                                        os.remove(res["tracked_video_path"])

                                    save_db(st.session_state.db)
                                    st.rerun()
                                finally:
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)

                        # Встроенный видеоплеер с трекингом ключевых точек для судьи
                        if entry["id"] in st.session_state.tracked_videos:
                            st.markdown("🎬 **Видеозапись с трекингом ключевых точек (MediaPipe Pose):**")
                            video_bytes = st.session_state.tracked_videos[entry["id"]]
                            st.video(video_bytes)
                            st.download_button(
                                label="💾 Скачать видео с трекингом (.mp4)",
                                data=video_bytes,
                                file_name=f"Tracking_{entry['school']}_{entry['fio']}.mp4",
                                mime="video/mp4",
                                key=f"dl_vid_{entry['id']}",
                                use_container_width=True
                            )

                        if entry.get("reps_detail"):
                            st.markdown("**Протокол повторений по таймкодам:**")
                            df_reps = pd.DataFrame(entry["reps_detail"])
                            df_reps["errors"] = df_reps["errors"].apply(lambda lst: ", ".join(lst))
                            df_reps = df_reps[["rep_number", "timestamp", "status_text", "errors"]]
                            df_reps.columns = ["№", "Таймкод", "Вердикт", "Причина / Ошибки"]
                            st.dataframe(df_reps, use_container_width=True, hide_index=True)

                    with col_right:
                        verified_val = st.number_input(
                            "Итоговое количество ЧИСТЫХ подтягиваний (в зачет):",
                            min_value=0, max_value=100,
                            value=int(entry["verified_reps"]) if entry["verified_reps"] is not None else int(entry["claimed_reps"]),
                            key=f"val_{entry['id']}"
                        )
                        judge_comment = st.text_area(
                            "Детальный лог по таймкодам и комментарий судьи:",
                            value=entry.get("comment", ""),
                            height=220,
                            key=f"comm_{entry['id']}"
                        )

                        b_save, b_del = st.columns([3, 1])
                        with b_save:
                            if st.button("💾 Утвердить в протокол", key=f"save_{entry['id']}", use_container_width=True):
                                entry["verified_reps"] = verified_val
                                entry["comment"] = judge_comment
                                entry["status"] = "Проверено"
                                save_db(st.session_state.db)
                                st.success("Результат утвержден!")
                                st.rerun()
                        with b_del:
                            if st.button("🗑️ Удалить", key=f"del_{entry['id']}", use_container_width=True):
                                st.session_state.db = [x for x in st.session_state.db if x["id"] != entry["id"]]
                                save_db(st.session_state.db)
                                st.rerun()
    else:
        st.error("Неверный пароль судьи.")
