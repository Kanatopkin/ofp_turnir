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
    page_title="Турнир по подтягиванию | Командный зачет",
    page_icon="🏆",
    layout="wide"
)

DB_FILENAME = "tournament_db.json"


# --- НАДЕЖНОЕ ПОДКЛЮЧЕНИЕ К GOOGLE DRIVE ---
def get_drive_service():
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


def delete_file_from_drive(file_id):
    """Безопасное удаление видеофайла из Google Drive при удалении участника или школы."""
    if not file_id:
        return
    try:
        service = get_drive_service()
        service.files().delete(fileId=file_id).execute(num_retries=2)
    except Exception:
        pass


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


def build_team_standings_df(db_records):
    """Формирует таблицу командного зачета по сумме 10 лучших результатов каждой школы."""
    if not db_records:
        return pd.DataFrame()

    df = pd.DataFrame(db_records)
    if "coach_fio" not in df.columns:
        df["coach_fio"] = "—"
    df["score"] = df["verified_reps"].fillna(0).astype(int)

    team_rows = []
    for sch, group in df.groupby("school"):
        total_m = len(group)
        checked_m = len(group[group["status"] == "Проверено"])
        top10_sum = int(group.sort_values(by="score", ascending=False).head(10)["score"].sum())
        coach_name = group["coach_fio"].iloc[-1] if "coach_fio" in group.columns else "—"
        team_rows.append({
            "Школа": sch,
            "Тренер (представитель)": coach_name,
            "Заявлено (10–12 чел.)": f"{total_m} чел.",
            "Проверено": f"{checked_m} из {total_m}",
            "Допуск команды (≥10 чел.)": "✅ Допущена" if total_m >= 10 else "⏳ Некомплект (<10)",
            "Сумма 10 лучших": top10_sum,
            "_is_eligible": 1 if total_m >= 10 else 0
        })

    df_teams = pd.DataFrame(team_rows).sort_values(
        by=["_is_eligible", "Сумма 10 лучших"],
        ascending=[False, False]
    ).reset_index(drop=True)
    df_teams.insert(0, "Место", range(1, len(df_teams) + 1))
    return df_teams.drop(columns=["_is_eligible"])


def build_school_roster_df(school_entries):
    """Формирует таблицу участников внутри конкретной школы с отметкой вхождения в Топ-10."""
    sorted_entries = sorted(
        school_entries,
        key=lambda x: (
            1 if x.get("status") == "Проверено" else 0,
            int(x["verified_reps"]) if x.get("verified_reps") is not None else -1,
            int(x.get("claimed_reps", 0))
        ),
        reverse=True
    )

    checked_ids_top10 = {
        x["id"] for x in sorted_entries
        if x.get("status") == "Проверено" and x.get("verified_reps") is not None
    }
    checked_ordered = [
        x["id"] for x in sorted_entries
        if x["id"] in checked_ids_top10
    ][:10]
    top10_set = set(checked_ordered)

    rows = []
    for idx, item in enumerate(sorted_entries, start=1):
        is_checked = item.get("status") == "Проверено"
        v_reps = item.get("verified_reps")

        if not is_checked or v_reps is None:
            top10_label = "⏳ Ожидание проверки"
            status_label = "🟡 На проверке"
            v_reps_display = "—"
        else:
            v_reps_display = str(int(v_reps))
            status_label = "🟢 Проверено"
            top10_label = "✅ В зачете Топ-10" if item["id"] in top10_set else "➖ Запасной (11–12)"

        err_list = item.get("errors", [])
        if is_checked:
            err_text = ", ".join(err_list) if err_list else "Без замечаний"
        else:
            err_text = "В очереди на проверку"

        rows.append({
            "№": idx,
            "ФИО участника": item.get("fio", "—"),
            "Класс": item.get("grade", "—"),
            "Заявлено": item.get("claimed_reps", 0),
            "Засчитано": v_reps_display,
            "Учет в Топ-10": top10_label,
            "Статус": status_label,
            "Примечание судейской системы": err_text
        })

    return pd.DataFrame(rows)


# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ СЕССИИ ---
if "db" not in st.session_state:
    try:
        st.session_state.db = load_db()
    except Exception as e:
        st.session_state.db = []
        st.warning(f"Ожидание настройки подключения к Google Drive в Secrets: {e}")

if "tracked_videos" not in st.session_state:
    st.session_state.tracked_videos = {}

if "active_school" not in st.session_state:
    st.session_state.active_school = None

# --- ШАПКА ПРИЛОЖЕНИЯ ---
st.title("🏆 Турнир по подтягиванию на перекладине памяти героев СВО")
st.markdown(
    "**Сроки проведения:** 20–27 октября | **Участники:** 8–9 классы (Московская область) | "
    "**Формат:** Командный зачет (состав команды 10–12 человек, зачет по сумме 10 лучших результатов)"
)

with st.expander("🤖 Архитектура компьютерного зрения (Pose Estimation) и критерии"):
    st.markdown("""
    * **Отслеживание ключевых точек (MediaPipe Pose):** кисти, локти, плечи, подбородок, тазобедренные суставы, колени и лодыжки.
    * **Контроль исходного положения (ИП):** вис хватом сверху, руки и ноги прямые, ступни вместе без скрещивания, обязательная фиксация неподвижного виса **не менее 1,0 секунды** перед началом каждого подъема.
    * **Контроль подъема и верхней точки:** пересечение подбородком горизонтальной линии грифа перекладины, равномерное сгибание рук без рывков и махов ногами (туловищем).
    """)

st.divider()

# =====================================================================
# БЛОК 1: РЕГИСТРАЦИЯ ШКОЛЫ (1 РАЗ) И ПОСЛЕДОВАТЕЛЬНОЕ ДОБАВЛЕНИЕ УЧАСТНИКОВ
# =====================================================================
col_reg, col_board = st.columns([1.0, 1.4], gap="large")

with col_reg:
    st.subheader("📥 Регистрация команды школы")

    # Если школа еще не зафиксирована в текущей сессии — показываем Шаг 1
    if st.session_state.active_school is None:
        st.markdown("##### Шаг 1. Данные школы и тренера (вводятся один раз)")

        existing_schools = sorted({x["school"] for x in st.session_state.db if x.get("school")})
        mode = "Зарегистрировать новую школу"
        if existing_schools:
            mode = st.radio(
                "Выберите режим:",
                ["Зарегистрировать новую школу", "Продолжить добавление в существующую школу"],
                horizontal=True
            )

        if mode == "Продолжить добавление в существующую школу" and existing_schools:
            sel_school = st.selectbox("Выберите школу из базы:", existing_schools)
            school_records = [x for x in st.session_state.db if x["school"] == sel_school]
            last_coach_fio = school_records[-1].get("coach_fio", "") if school_records else ""
            last_coach_phone = school_records[-1].get("coach_phone", "") if school_records else ""

            with st.form("select_existing_school_form"):
                coach_fio = st.text_input("ФИО тренера (представителя команды)*", value=last_coach_fio).strip()
                coach_phone = st.text_input("Номер телефона тренера*", value=last_coach_phone, placeholder="+7 (999) 000-00-00").strip()
                lock_btn = st.form_submit_button("🔒 Перейти к добавлению участников школы", use_container_width=True)
                if lock_btn:
                    if not coach_fio or not coach_phone:
                        st.error("Укажите ФИО и контактный телефон тренера.")
                    else:
                        st.session_state.active_school = {
                            "school": sel_school,
                            "coach_fio": coach_fio,
                            "coach_phone": coach_phone
                        }
                        st.rerun()
        else:
            with st.form("school_setup_form"):
                school_name = st.text_input("Наименование школы (Московская область)*", placeholder="МБОУ СОШ №1 г. ...").strip()
                coach_fio = st.text_input("ФИО тренера (представителя команды)*", placeholder="Петров Пётр Петрович").strip()
                coach_phone = st.text_input("Номер телефона тренера*", placeholder="+7 (999) 123-45-67").strip()
                lock_btn = st.form_submit_button("🔒 Зафиксировать школу и добавить участников", use_container_width=True)

                if lock_btn:
                    if not school_name or not coach_fio or not coach_phone:
                        st.error("Заполните наименование школы, ФИО тренера и номер телефона.")
                    else:
                        st.session_state.active_school = {
                            "school": school_name,
                            "coach_fio": coach_fio,
                            "coach_phone": coach_phone
                        }
                        st.rerun()

    # Если школа зафиксирована — показываем карточку школы и Шаг 2 (добавление участников)
    else:
        act = st.session_state.active_school
        school_entries = [
            x for x in st.session_state.db
            if x["school"].lower() == act["school"].lower()
        ]
        count_added = len(school_entries)

        st.info(
            f"🏫 **Школа:** {act['school']}  \n"
            f"👨‍🏫 **Тренер:** {act['coach_fio']} ({act['coach_phone']})  \n"
            f"👥 **Участников в команде:** **{count_added} из 12** "
            f"{'(✅ Командный минимум 10 чел. выполнен)' if count_added >= 10 else f'(Для зачета нужно еще мин. {10 - count_added} чел.)'}"
        )

        if st.button("✏️ Сменить школу / редактировать данные тренера", use_container_width=True):
            st.session_state.active_school = None
            st.rerun()

        st.markdown("##### Шаг 2. Добавление участника в команду школы")
        if count_added >= 12:
            st.warning(f"Достигнут максимальный лимит для команды школы «{act['school']}» (12 участников).")
        else:
            with st.form("add_participant_form", clear_on_submit=True):
                fio = st.text_input("ФИО участника*", placeholder="Иванов Иван Иванович").strip()
                col_g, col_r = st.columns(2)
                with col_g:
                    grade = st.selectbox("Класс*", ["8 класс", "9 класс"])
                with col_r:
                    self_reps = st.number_input("Заявлено подтягиваний", min_value=0, max_value=100, value=10)

                video_file = st.file_uploader("Видеозапись выполнения (MP4, MOV)*", type=["mp4", "mov", "avi", "mkv"])
                submitted = st.form_submit_button(
                    f"➕ Добавить участника №{count_added + 1} в команду",
                    use_container_width=True
                )

                if submitted:
                    if not fio or not video_file:
                        st.error("Укажите ФИО участника и прикрепите видеофайл.")
                    else:
                        with st.spinner(f"Загрузка видео участника {fio} в Google Drive..."):
                            safe_name = f"{act['school']}_{grade}_{fio}_{datetime.now().strftime('%m%d_%H%M')}.mp4"
                            file_id, web_link = upload_video_to_drive(video_file, safe_name)
                            new_entry = {
                                "id": f"id_{int(time.time()*1000)}",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "school": act["school"],
                                "coach_fio": act["coach_fio"],
                                "coach_phone": act["coach_phone"],
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
                            st.success(f"Участник {fio} добавлен в команду «{act['school']}»!")
                            st.rerun()

        if school_entries:
            with st.expander(f"📋 Текущий состав команды «{act['school']}» ({count_added} чел.)", expanded=True):
                st.dataframe(build_school_roster_df(school_entries), use_container_width=True, hide_index=True)

# =====================================================================
# БЛОК 2: ОНЛАЙН-ТАБЛО КОМАНДНОГО ЗАЧЕТА + РАСКРЫВАЮЩИЕСЯ СОСТАВЫ ШКОЛ
# =====================================================================
with col_board:
    col_title, col_sync = st.columns([3, 1])
    with col_title:
        st.subheader("🏫 Командный зачет (Сумма 10 лучших)")
    with col_sync:
        if st.button("🔄 Обновить", use_container_width=True):
            st.session_state.db = load_db()
            st.rerun()

    if not st.session_state.db:
        st.info("Заявки от команд школ пока не поступали.")
    else:
        df_teams = build_team_standings_df(st.session_state.db)
        st.dataframe(df_teams, use_container_width=True, hide_index=True)
        st.caption(
            "💡 Итоговый результат команды определяется по сумме 10 лучших результатов участников школы "
            "(состав команды: не менее 10 и не более 12 человек)."
        )

        st.markdown("##### 📋 Составы команд и статус проверки участников по школам")
        for _, team_row in df_teams.iterrows():
            rank = team_row["Место"]
            sch_name = team_row["Школа"]
            top10_val = team_row["Сумма 10 лучших"]
            checked_info = team_row["Проверено"]
            coach_str = team_row["Тренер (представитель)"]

            sch_entries = [x for x in st.session_state.db if x.get("school") == sch_name]
            with st.expander(
                f"Место #{rank} | 🏫 {sch_name} — Сумма Топ-10: {top10_val} | "
                f"Проверено: {checked_info} | Тренер: {coach_str}"
            ):
                df_roster = build_school_roster_df(sch_entries)
                st.dataframe(df_roster, use_container_width=True, hide_index=True)

st.divider()

# =====================================================================
# БЛОК 3: СУДЕЙСКИЙ МОДУЛЬ С ГРУППИРОВКОЙ ПО ШКОЛАМ И ПАКЕТНЫМ АНАЛИЗОМ
# =====================================================================
st.subheader("⚖️ Судейский модуль (Группировка по школам и поочередный Pose-анализ)")
judge_pwd = st.text_input("Введите пароль судьи для доступа к проверке команд и управлению протоколами:", type="password")

if judge_pwd:
    if judge_pwd == st.secrets["general"]["judge_password"]:
        st.success("Авторизация судьи выполнена успешно.")

        if not st.session_state.db:
            st.info("Нет загруженных заявок для судейства.")
        else:
            c_filt, c_exp = st.columns([2, 1])
            with c_filt:
                filter_status = st.radio(
                    "Фильтр участников внутри школ:",
                    ["Все", "На проверке", "Проверено"],
                    horizontal=True
                )
            with c_exp:
                df_teams_exp = build_team_standings_df(st.session_state.db)
                df_details_exp = pd.DataFrame(st.session_state.db).drop(columns=["reps_detail"], errors="ignore")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_teams_exp.to_excel(writer, index=False, sheet_name="Командный зачет")
                    df_details_exp.to_excel(writer, index=False, sheet_name="Участники по школам")
                st.download_button(
                    "📥 Скачать протокол командного зачета (Excel)",
                    data=buf.getvalue(),
                    file_name=f"Командный_протокол_Подтягивание_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            # Группируем все записи по школам
            schools_list = sorted({x["school"] for x in st.session_state.db if x.get("school")})

            for sch_idx, school_name in enumerate(schools_list):
                all_school_entries = [x for x in st.session_state.db if x.get("school") == school_name]
                unchecked_entries = [x for x in all_school_entries if x.get("status") != "Проверено"]
                checked_count = len(all_school_entries) - len(unchecked_entries)

                coach_fio_val = all_school_entries[-1].get("coach_fio", "Не указан")
                coach_phone_val = all_school_entries[-1].get("coach_phone", "Не указан")

                scores_sorted = sorted(
                    [int(x["verified_reps"]) for x in all_school_entries if x.get("verified_reps") is not None],
                    reverse=True
                )
                school_top10_sum = sum(scores_sorted[:10])

                with st.expander(
                    f"🏫 {school_name} | Участников: {len(all_school_entries)} (Проверено: {checked_count}/{len(all_school_entries)}) | "
                    f"Командная сумма (Топ-10): {school_top10_sum} | Тренер: {coach_fio_val} ({coach_phone_val})",
                    expanded=(len(unchecked_entries) > 0)
                ):
                    st.markdown(
                        f"**Представитель (тренер):** {coach_fio_val} | **Телефон:** `{coach_phone_val}` | "
                        f"**Статус командного зачета:** {'✅ Допущена (≥10 чел.)' if len(all_school_entries) >= 10 else f'⏳ Некомплект ({len(all_school_entries)} из 10)'}"
                    )

                    # Кнопки пакетного запуска анализа и удаления всей команды школы
                    b_batch_new, b_batch_all, b_del_school = st.columns([1.35, 1.15, 0.85])
                    run_batch_target = None

                    with b_batch_new:
                        if st.button(
                            f"🚀 Анализ + видео непроверенных ({len(unchecked_entries)} чел.)",
                            key=f"batch_new_{sch_idx}",
                            disabled=(len(unchecked_entries) == 0),
                            use_container_width=True
                        ):
                            run_batch_target = unchecked_entries

                    with b_batch_all:
                        if st.button(
                            f"🔄 Пересчитать ВСЮ школу ({len(all_school_entries)} чел.)",
                            key=f"batch_all_{sch_idx}",
                            use_container_width=True
                        ):
                            run_batch_target = all_school_entries

                    with b_del_school:
                        with st.popover("🗑️ Удалить всю школу", use_container_width=True):
                            st.warning(
                                f"Вы действительно хотите полностью удалить команду **«{school_name}»** "
                                f"({len(all_school_entries)} чел.) и все её видеозаписи?"
                            )
                            if st.button(
                                f"❌ Да, удалить «{school_name}»",
                                key=f"confirm_del_school_{sch_idx}",
                                type="primary",
                                use_container_width=True
                            ):
                                with st.spinner(f"Удаление команды «{school_name}» и очистка файлов..."):
                                    for rem_item in all_school_entries:
                                        delete_file_from_drive(rem_item.get("file_id"))
                                        st.session_state.tracked_videos.pop(rem_item["id"], None)

                                    st.session_state.db = [
                                        x for x in st.session_state.db
                                        if x.get("school") != school_name
                                    ]
                                    if (
                                        st.session_state.active_school
                                        and st.session_state.active_school.get("school") == school_name
                                    ):
                                        st.session_state.active_school = None

                                    save_db(st.session_state.db)
                                st.success(f"Команда школы «{school_name}» полностью удалена!")
                                st.rerun()

                    # ПООЧЕРЕДНОЕ ВЫПОЛНЕНИЕ АНАЛИЗА И ГЕНЕРАЦИИ ВИДЕО ДЛЯ ШКОЛЫ
                    if run_batch_target:
                        total_q = len(run_batch_target)
                        overall_bar = st.progress(0.0, text=f"Подготовка к поочередной обработке школы «{school_name}» (0/{total_q})...")
                        frame_bar = st.progress(0.0, text="Ожидание первого видео...")
                        status_box = st.empty()

                        for idx_p, item in enumerate(run_batch_target, start=1):
                            overall_bar.progress(
                                (idx_p - 1) / total_q,
                                text=f"🏫 Школа «{school_name}»: обработка участника {idx_p} из {total_q} — {item['fio']} ({item['grade']})"
                            )
                            status_box.info(f"⏳ Скачивание и покадровый трекинг: **{item['fio']}** ({idx_p}/{total_q})...")

                            tmp_path = download_video_temp(item["file_id"])
                            try:
                                manual_bar = float(item["bar_y"]) if item.get("bar_y") is not None else None
                                res = analyze_video_pose(
                                    tmp_path,
                                    manual_bar_y=manual_bar,
                                    generate_tracked_video=True,
                                    progress_callback=lambda p, fio=item["fio"], i=idx_p: frame_bar.progress(
                                        p, text=f"🎥 [{i}/{total_q}] {fio} — трекинг ключевых точек: {int(p*100)}%"
                                    )
                                )
                                if isinstance(res, dict):
                                    item["verified_reps"] = res.get("valid_reps", 0)
                                    item["bar_y"] = res.get("bar_y", 0.12)
                                    item["errors"] = res.get("errors", [])
                                    item["reps_detail"] = res.get("reps_detail", [])
                                    item["comment"] = res.get("comment", "")
                                    item["status"] = "Проверено"

                                    tracked_path = res.get("tracked_video_path")
                                    if tracked_path and os.path.exists(tracked_path):
                                        with open(tracked_path, "rb") as vf:
                                            st.session_state.tracked_videos[item["id"]] = vf.read()
                                        os.remove(tracked_path)

                                    save_db(st.session_state.db)
                            except Exception as e:
                                st.error(f"Ошибка при обработке участника {item['fio']}: {e}")
                            finally:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)

                        overall_bar.progress(1.0, text=f"✅ Все участники школы «{school_name}» ({total_q} чел.) успешно обработаны!")
                        frame_bar.empty()
                        status_box.success(f"Пакетный анализ школы «{school_name}» завершен! Все видео с трекингом доступны ниже.")
                        time.sleep(0.8)
                        st.rerun()

                    st.divider()

                    # Фильтрация участников внутри выбранной школы
                    school_filtered = [
                        x for x in all_school_entries
                        if filter_status == "Все" or x.get("status") == filter_status
                    ]

                    if not school_filtered:
                        st.caption("В этой школе нет участников, соответствующих выбранному фильтру.")

                    # Индивидуальные карточки участников внутри школы
                    for entry in school_filtered:
                        with st.expander(
                            f"{'🟢' if entry['status']=='Проверено' else '🟡'} {entry['fio']} ({entry['grade']}) | "
                            f"Заявлено: {entry['claimed_reps']} | Засчитано в зачет школы: {entry['verified_reps'] if entry['verified_reps'] is not None else '—'}"
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
                                        value=float(entry["bar_y"]) if entry.get("bar_y") else 0.12,
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
                                                    st.image(img, caption=f"Линия грифа (авто-определение: Y={auto_y})", use_container_width=True)
                                            finally:
                                                if os.path.exists(tmp_path):
                                                    os.remove(tmp_path)

                                with b_run:
                                    if st.button("🤖 Анализ + Видео (индив.)", key=f"pose_{entry['id']}", use_container_width=True):
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

                                            if isinstance(res, dict):
                                                entry["verified_reps"] = res.get("valid_reps", 0)
                                                entry["bar_y"] = res.get("bar_y", 0.12)
                                                entry["errors"] = res.get("errors", [])
                                                entry["reps_detail"] = res.get("reps_detail", [])
                                                entry["comment"] = res.get("comment", "")

                                                tracked_path = res.get("tracked_video_path")
                                                if tracked_path and os.path.exists(tracked_path):
                                                    with open(tracked_path, "rb") as vf:
                                                        st.session_state.tracked_videos[entry["id"]] = vf.read()
                                                    os.remove(tracked_path)

                                                save_db(st.session_state.db)
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"Ошибка при выполнении анализа видео: {type(e).__name__}: {e}")
                                        finally:
                                            if os.path.exists(tmp_path):
                                                os.remove(tmp_path)

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
                                    df_reps["errors"] = df_reps["errors"].apply(
                                        lambda lst: ", ".join(lst) if isinstance(lst, list) else str(lst)
                                    )
                                    df_reps = df_reps[["rep_number", "timestamp", "status_text", "errors"]]
                                    df_reps.columns = ["№", "Таймкод", "Вердикт", "Причина / Ошибки"]
                                    st.dataframe(df_reps, use_container_width=True, hide_index=True)

                            with col_right:
                                verified_val = st.number_input(
                                    "Итоговое количество ЧИСТЫХ подтягиваний (в зачет команды):",
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
                                    if st.button("💾 Утвердить в командный протокол", key=f"save_{entry['id']}", use_container_width=True):
                                        entry["verified_reps"] = verified_val
                                        entry["comment"] = judge_comment
                                        entry["status"] = "Проверено"
                                        save_db(st.session_state.db)
                                        st.success("Результат утвержден!")
                                        st.rerun()
                                with b_del:
                                    if st.button("🗑️ Удалить", key=f"del_{entry['id']}", use_container_width=True):
                                        delete_file_from_drive(entry.get("file_id"))
                                        st.session_state.tracked_videos.pop(entry["id"], None)
                                        st.session_state.db = [x for x in st.session_state.db if x["id"] != entry["id"]]
                                        save_db(st.session_state.db)
                                        st.rerun()
    else:
        st.error("Неверный пароль судьи.")
