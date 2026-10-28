import io
import os
import json
import base64
import tempfile
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image

# Импорт аналитического движка компьютерного зрения
try:
    from pose_analyzer import analyze_video_pose, get_calibration_preview, CONFIG
except ImportError as e:
    st.error(f"Ошибка импорта pose_analyzer: {e}")

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Турнир по подтягиванию памяти героев СВО",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пути к локальным базам данных (для автономной работы без облака)
DATA_DIR = os.path.join(os.path.dirname(__file__), "tournament_data")
os.makedirs(DATA_DIR, exist_ok=True)
TEAMS_DB_PATH = os.path.join(DATA_DIR, "teams_database.json")

# Инициализация хранилища данных команд
def load_teams_db():
    if os.path.exists(TEAMS_DB_PATH):
        try:
            with open(TEAMS_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_teams_db(data):
    try:
        with open(TEAMS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Ошибка сохранения локальной базы: {e}")

# Инициализация подключения к Google Drive
def init_google_drive():
    try:
        # 1. Проверка Service Account
        if "gcp_service_account" in st.secrets:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            creds = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            return build('drive', 'v3', credentials=creds), None
        
        # 2. Проверка пользовательского OAuth refresh_token
        if "google_drive" in st.secrets:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            gd = st.secrets["google_drive"]
            creds = Credentials(
                token=None,
                refresh_token=gd.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=gd.get("client_id"),
                client_secret=gd.get("client_secret"),
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            return build('drive', 'v3', credentials=creds), None
            
        return None, "Секреты Google Drive не настроены (данные сохраняются локально)"
    except Exception as e:
        return None, f"('invalid_grant: Token has been expired or revoked.', {{'error': 'invalid_grant', 'error_description': '{e}'}})"

drive_service, drive_error = init_google_drive()

# Баннер предупреждения Google Drive
if drive_error:
    st.warning(f"Ожидание настройки подключения к Google Drive в Secrets: {drive_error}")

# Заголовок турнира
st.markdown("""
<h1 style='margin-bottom: 0px;'>🏆 Турнир по подтягиванию на перекладине памяти героев СВО</h1>
<p style='color: #444; font-size: 1.05rem; margin-top: 4px; margin-bottom: 12px;'>
<b>Сроки проведения:</b> 20–27 октября | <b>Участники:</b> 8–9 классы (Московская область) | <b>Формат:</b> Командный зачет (состав команды 10–12 человек, зачет по сумме 10 лучших результатов)
</p>
""", unsafe_allow_html=True)

# Описание правил и гибридной архитектуры
with st.expander("ℹ️ Гибридная архитектура судейства (YOLOv8-Pose + ИИ-арбитр Gemini) и критерии"):
    st.markdown("""
    * **Компьютерное зрение (YOLOv8-Pose):** Покадровый расчет ключевых суставов (углы сгибания в локтях и коленях, фиксация линии грифа, наклон головы и плеч).
    * **Независимый ИИ-арбитр (Gemini API):** Беспристрастный мультимодальный аудит сторибордов пиковых фаз подтягивания по нормативам ВсОШ и ГТО.
    * **Требования регламента:** 
        1. Исходное положение (ИП): неподвижный чистый вис на прямых руках не менее **0.72 с**.
        2. Верхняя точка: подбородок поднят строго **выше уровня перекладины**.
        3. Запрещено: раскачивание и киппинг (мах ногами), сгибание ног в коленях, перекрещивание стоп, поочередное подтягивание на одну руку и закидывание головы назад.
    """)

# Основная рабочая область в 2 колонки
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader("🎽 Регистрация команды школы")
    
    # Шаг 1. Данные школы
    st.markdown("##### Шаг 1. Данные школы и тренера (вводятся один раз)")
    school_name = st.text_input(
        "Наименование школы (Московская область)*",
        placeholder="МБОУ СОШ №1 г. ...",
        key="school_name_input"
    )
    coach_fio = st.text_input(
        "ФИО тренера / ответственного преподавателя*",
        placeholder="Иванов Иван Иванович",
        key="coach_fio_input"
    )

    st.markdown("---")

    # Шаг 2. Данные участника
    st.markdown("##### Шаг 2. Данные участника")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        athlete_fio = st.text_input("ФИО участника*", placeholder="Смирнов Алексей", key="athlete_fio")
    with col_u2:
        athlete_class = st.selectbox("Класс*", ["8 класс", "9 класс"], key="athlete_class")

    # Шаг 3. Загрузка видеозаписи
    st.markdown("##### Шаг 3. Видеозапись выступления")
    uploaded_file = st.file_uploader(
        "Загрузите видео подтягиваний (.mp4, .mov, .avi)",
        type=["mp4", "mov", "avi"],
        help="Рекомендуется боковая или диагональная съёмка со стабильного штатива с полной видимостью спортсмена и перекладины."
    )

    if uploaded_file is not None:
        # Сохранение видео во временный файл
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
        tfile.close()

        # Предпросмотр и калибровка грифа
        with st.spinner("Калибровка уровня перекладины по начальным кадрам..."):
            preview_img, auto_bar_y = get_calibration_preview(temp_video_path)

        st.markdown("###### Калибровка перекладины")
        if preview_img is not None:
            st.image(preview_img, caption="Контрольный кадр: зеленая линия — верхний уровень грифа", use_container_width=True)

        manual_bar_y = st.slider(
            "Точная подстройка высоты перекладины (Y-координата)",
            min_value=0.05,
            max_value=0.35,
            value=float(auto_bar_y),
            step=0.005,
            help="Передвиньте ползунок, чтобы зеленая линия строго совпала с верхней поверхностью грифа."
        )

        force_ai = st.checkbox(
            "Включить полную независимую проверку каждого повторения через Gemini API",
            value=False,
            help="По умолчанию локальное компьютерное зрение производит оценку мгновенно. Данная опция подключает облачного ИИ-арбитра ко всем попыткам."
        )

        # Кнопка запуска анализа
        if st.button("🚀 Начать судейство и сформировать протокол", type="primary", use_container_width=True):
            if not school_name or not athlete_fio:
                st.error("Пожалуйста, заполните наименование школы и ФИО участника перед началом анализа.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(val):
                    progress_bar.progress(val)
                    status_text.text(f"Выполняется биомеханический анализ попытки: {int(val*100)}%")

                with st.spinner("Идет покадровый анализ движения скелета нейросетью..."):
                    results = analyze_video_pose(
                        video_path=temp_video_path,
                        manual_bar_y=manual_bar_y,
                        force_ai_all_reps=force_ai,
                        progress_callback=update_progress
                    )

                progress_bar.progress(1.0)
                status_text.success("Судейская экспертиза успешно завершена!")

                # Сохраняем результат в сессии
                st.session_state["last_analysis_results"] = results
                st.session_state["current_athlete"] = {
                    "school": school_name,
                    "coach": coach_fio,
                    "athlete": athlete_fio,
                    "class": athlete_class,
                    "valid_reps": results["valid_reps"],
                    "total_attempts": results["total_attempts"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }

    # Отображение результатов последнего анализа
    if "last_analysis_results" in st.session_state:
        results = st.session_state["last_analysis_results"]
        ath_info = st.session_state.get("current_athlete", {})

        st.markdown("---")
        st.subheader("📊 Результаты судейской экспертизы")

        # Метрики зачета
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Засчитано (ГТО/ВсОШ)", f"{results['valid_reps']} повт.")
        m_col2.metric("Всего попыток", f"{results['total_attempts']}")
        invalid_count = results['total_attempts'] - results['valid_reps']
        m_col3.metric("Забраковано", f"{invalid_count}")

        # Видеозапись с трекингом и HUD
        if results.get("tracked_video_path") and os.path.exists(results["tracked_video_path"]):
            st.markdown("###### Видеозапись попытки с наложенной телеметрией и судейским HUD:")
            st.video(results["tracked_video_path"])

        # Формирование детальной таблицы протокола
        export_rows = []
        for r in results.get("reps_detail", []):
            errors_formatted = "; ".join(r["errors"]) if r["errors"] else "Нет ошибок (чистое выполнение)"
            export_rows.append({
                "№": r["rep_number"],
                "Таймкод": r["timestamp"],
                "Вердикт": r["status_text"],
                "Индикатор ИИ-арбитра": r.get("ai_indicator", "⚖️ Локальный FSM"),
                "Заключение ИИ": r.get("ai_verdict", "—"),
                "Причина / Ошибки": errors_formatted,
            })

        df_export = pd.DataFrame(export_rows)

        st.markdown("###### Детализированный судейский протокол по каждому повторению:")
        st.dataframe(df_export, use_container_width=True, hide_index=True)

        # Выгрузка отчетов (CSV и Excel)
        st.markdown("###### Выгрузка протокола:")
        btn_col1, btn_col2 = st.columns(2)

        # 1. Скачать CSV с правильной кодировкой для Excel (utf-8-sig)
        csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        filename_base = f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_{ath_info.get('athlete', 'export')}"
        btn_col1.download_button(
            label="📥 Скачать протокол (CSV)",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 2. Скачать Excel с безопасной обработкой openpyxl
        excel_buffer = io.BytesIO()
        excel_ready = False
        try:
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Протокол")
            excel_ready = True
        except Exception:
            excel_ready = False

        if excel_ready:
            btn_col2.download_button(
                label="📊 Скачать протокол (Excel .xlsx)",
                data=excel_buffer.getvalue(),
                file_name=f"{filename_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            btn_col2.info("Для экспорта .xlsx установите: pip install openpyxl")

        # Кнопка внесения результата в командный зачет школы
        if st.button("💾 Зафиксировать результат в командном зачете школы", type="secondary", use_container_width=True):
            teams_db = load_teams_db()
            s_name = ath_info.get("school", "Школа")
            if s_name not in teams_db:
                teams_db[s_name] = {
                    "coach": ath_info.get("coach", "—"),
                    "participants": []
                }
            
            # Проверка на дублирование участника
            existing_idx = None
            for idx, p in enumerate(teams_db[s_name]["participants"]):
                if p["fio"] == ath_info.get("athlete") and p["class"] == ath_info.get("class"):
                    existing_idx = idx
                    break
            
            p_entry = {
                "fio": ath_info.get("athlete"),
                "class": ath_info.get("class"),
                "valid_reps": ath_info.get("valid_reps"),
                "total_attempts": ath_info.get("total_attempts"),
                "date": ath_info.get("timestamp")
            }

            if existing_idx is not None:
                teams_db[s_name]["participants"][existing_idx] = p_entry
                st.success(f"Результат участника {ath_info.get('athlete')} успешно обновлен: {ath_info.get('valid_reps')} подтягиваний!")
            else:
                teams_db[s_name]["participants"].append(p_entry)
                st.success(f"Участник {ath_info.get('athlete')} добавлен в состав команды {s_name}!")

            save_teams_db(teams_db)
            st.rerun()

with col2:
    st.subheader("🏫 Командный зачет (Сумма 10 лучших)")
    teams_db = load_teams_db()

    if not teams_db:
        st.info("Заявки от команд школ пока не поступали.")
    else:
        # Формирование рейтинга школ
        school_rankings = []
        for s_name, s_data in teams_db.items():
            participants = s_data.get("participants", [])
            # Сортировка участников по убыванию засчитанных подтягиваний
            sorted_p = sorted(participants, key=lambda x: x["valid_reps"], reverse=True)
            top_10 = sorted_p[:10]
            team_score = sum(p["valid_reps"] for p in top_10)
            school_rankings.append({
                "school": s_name,
                "coach": s_data.get("coach", "—"),
                "count": len(participants),
                "top10_score": team_score,
                "top_participants": sorted_p
            })

        # Сортировка школ по сумме 10 лучших
        school_rankings.sort(key=lambda x: x["top10_score"], reverse=True)

        for rank, s_info in enumerate(school_rankings, 1):
            with st.expander(f"🏅 {rank} место: {s_info['school']} — {s_info['top10_score']} баллов ({s_info['count']} участников)", expanded=(rank == 1)):
                st.markdown(f"**Тренер/Представитель:** {s_info['coach']}")
                
                table_rows = []
                for p_idx, p in enumerate(s_info["top_participants"], 1):
                    is_top10 = "⭐ В зачете" if p_idx <= 10 else "— Запас"
                    table_rows.append({
                        "№": p_idx,
                        "Участник": p["fio"],
                        "Класс": p["class"],
                        "Засчитано": f"{p['valid_reps']} повт.",
                        "Всего попыток": p["total_attempts"],
                        "Статус": is_top10,
                        "Дата": p.get("date", "—")
                    })
                
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
