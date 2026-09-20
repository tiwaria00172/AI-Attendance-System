import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import re
import urllib.request
from pathlib import Path
from datetime import datetime, date
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Attendance System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "student_photos"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

# IMPORTANT:
# Using the lightweight INT8 SFace model
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec_int8bq.onnx"

STUDENTS_FILE = DATA_DIR / "students.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"


# ============================================================
# MODEL URLS
# ============================================================

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec_int8bq.onnx"
)


# ============================================================
# CONSTANTS
# ============================================================

ATTENDANCE_COLUMNS = [
    "date",
    "time",
    "student_id",
    "name",
]

# Cosine similarity threshold
MATCH_THRESHOLD = 0.45

# Maximum image width for processing
MAX_IMAGE_WIDTH = 960


# ============================================================
# INITIAL FILE CREATION
# ============================================================

def initialize_files():
    """
    Creates all required files if they don't exist.
    Also repairs malformed attendance CSV files.
    """

    # -------------------------
    # Students JSON
    # -------------------------
    if not STUDENTS_FILE.exists():
        with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4)

    # -------------------------
    # Attendance CSV
    # -------------------------
    if not ATTENDANCE_FILE.exists():
        df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
        df.to_csv(ATTENDANCE_FILE, index=False)
    else:
        repair_attendance_csv()


def repair_attendance_csv():
    """
    Makes attendance.csv safe to use.

    Prevents errors such as:
    KeyError: 'date'
    """

    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except Exception:
        df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)

    # If CSV has no columns
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)

    # Add missing columns
    for column in ATTENDANCE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # Keep only expected columns
    df = df[ATTENDANCE_COLUMNS]

    # Convert values safely
    for column in ATTENDANCE_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    df.to_csv(ATTENDANCE_FILE, index=False)


initialize_files()


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_model(url, destination):
    """
    Downloads a model only if it does not already exist.
    """

    if destination.exists() and destination.stat().st_size > 10000:
        return True

    try:
        with st.spinner(f"Downloading {destination.name}..."):
            urllib.request.urlretrieve(url, destination)

        if destination.exists() and destination.stat().st_size > 10000:
            return True

        return False

    except Exception as e:
        st.error(f"Could not download {destination.name}: {e}")
        return False


def ensure_models():
    """
    Makes sure required ONNX models exist.
    """

    yunet_ok = download_model(
        YUNET_URL,
        YUNET_MODEL
    )

    sface_ok = download_model(
        SFACE_URL,
        SFACE_MODEL
    )

    return yunet_ok and sface_ok


# ============================================================
# LOAD AI MODELS
# ============================================================

@st.cache_resource
def load_models():
    """
    Loads YuNet and SFace once.

    Models remain cached while Streamlit is running.
    """

    if not ensure_models():
        raise RuntimeError(
            "One or more AI models could not be downloaded."
        )

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL),
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(SFACE_MODEL),
        ""
    )

    return detector, recognizer


# ============================================================
# IMAGE HELPERS
# ============================================================

def resize_image(image, max_width=MAX_IMAGE_WIDTH):
    """
    Resize image while maintaining aspect ratio.
    """

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def bytes_to_cv2(uploaded_file):
    """
    Convert Streamlit uploaded file/camera image to OpenCV image.
    """

    if uploaded_file is None:
        return None

    try:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return None

        return resize_image(image)

    except Exception:
        return None


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image, detector):
    """
    Detect faces using YuNet.
    """

    if image is None:
        return []

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        return []

    return faces


# ============================================================
# FACE EMBEDDING
# ============================================================

def get_embedding(image, face, recognizer):
    """
    Generates a normalized SFace embedding.
    """

    try:
        aligned_face = recognizer.alignCrop(
            image,
            face
        )

        feature = recognizer.feature(
            aligned_face
        )

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).flatten()

        norm = np.linalg.norm(feature)

        if norm == 0:
            return None

        feature = feature / norm

        return feature

    except Exception:
        return None


# ============================================================
# STUDENT DATABASE
# ============================================================

def load_students():
    """
    Load student information from JSON.
    """

    try:
        if not STUDENTS_FILE.exists():
            return {}

        with open(
            STUDENTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


def save_students(students):
    """
    Save student information.
    """

    with open(
        STUDENTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            students,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# EMBEDDINGS DATABASE
# ============================================================

def load_embeddings():
    """
    Load saved embeddings.

    Returns:
        dict:
            student_id -> numpy embedding
    """

    if not EMBEDDINGS_FILE.exists():
        return {}

    try:
        data = np.load(
            EMBEDDINGS_FILE,
            allow_pickle=False
        )

        embeddings = {}

        for key in data.files:
            embeddings[key] = data[key].astype(
                np.float32
            )

        return embeddings

    except Exception:
        return {}


def save_embeddings(embeddings):
    """
    Save embeddings in compressed NPZ format.
    """

    if not embeddings:
        # Create empty file
        np.savez_compressed(
            EMBEDDINGS_FILE
        )
        return

    np.savez_compressed(
        EMBEDDINGS_FILE,
        **embeddings
    )


# ============================================================
# ID SANITIZATION
# ============================================================

def safe_id(value):
    """
    Makes a safe ID for filenames / NPZ keys.
    """

    value = str(value).strip()

    value = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        value
    )

    return value


# ============================================================
# FACE MATCHING
# ============================================================

def find_best_match(
    embedding,
    embeddings,
    students
):
    """
    Finds the student with the highest cosine similarity.
    """

    if embedding is None:
        return None, 0.0

    if not embeddings:
        return None, 0.0

    best_student = None
    best_score = -1.0

    for student_id, stored_embedding in embeddings.items():

        if student_id not in students:
            continue

        try:
            stored_embedding = np.asarray(
                stored_embedding,
                dtype=np.float32
            ).flatten()

            current_embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).flatten()

            if stored_embedding.shape != current_embedding.shape:
                continue

            score = float(
                np.dot(
                    current_embedding,
                    stored_embedding
                )
            )

            if score > best_score:
                best_score = score
                best_student = student_id

        except Exception:
            continue

    if best_score >= MATCH_THRESHOLD:
        return best_student, best_score

    return None, best_score


# ============================================================
# ATTENDANCE FUNCTIONS
# ============================================================

def load_attendance():
    """
    Safely loads attendance.csv.

    This function is the main protection against:
        KeyError: 'date'
    """

    try:
        if not ATTENDANCE_FILE.exists():
            df = pd.DataFrame(
                columns=ATTENDANCE_COLUMNS
            )
            df.to_csv(
                ATTENDANCE_FILE,
                index=False
            )
            return df

        df = pd.read_csv(
            ATTENDANCE_FILE
        )

    except Exception:
        df = pd.DataFrame(
            columns=ATTENDANCE_COLUMNS
        )

    # Handle completely empty CSV
    if len(df.columns) == 0:
        df = pd.DataFrame(
            columns=ATTENDANCE_COLUMNS
        )

    # Add missing columns
    for column in ATTENDANCE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # Remove unexpected columns
    df = df[ATTENDANCE_COLUMNS]

    # Prevent NaN issues
    for column in ATTENDANCE_COLUMNS:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )

    return df


def save_attendance(df):
    """
    Save attendance safely.
    """

    # Ensure all required columns exist
    for column in ATTENDANCE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[ATTENDANCE_COLUMNS]

    df.to_csv(
        ATTENDANCE_FILE,
        index=False
    )


def already_marked_today(student_id):
    """
    Checks whether student already has attendance today.
    """

    df = load_attendance()

    if df.empty:
        return False

    today = str(date.today())

    matches = df[
        (df["date"] == today) &
        (df["student_id"] == str(student_id))
    ]

    return not matches.empty


def mark_attendance(student_id, name):
    """
    Mark attendance for a student.

    Only one attendance entry per student per day.
    """

    df = load_attendance()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    # Prevent duplicate attendance
    existing = df[
        (df["date"].astype(str) == today) &
        (df["student_id"].astype(str) == str(student_id))
    ]

    if not existing.empty:
        return False, "Already marked today"

    new_row = pd.DataFrame([
        {
            "date": today,
            "time": current_time,
            "student_id": str(student_id),
            "name": str(name),
        }
    ])

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    save_attendance(df)

    return True, "Attendance marked"


# ============================================================
# DRAW FACE BOX
# ============================================================

def draw_face(
    image,
    face,
    label,
    score=None
):
    """
    Draw bounding box and label.
    """

    x, y, w, h = face[:4].astype(int)

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    if score is not None:
        text = f"{label} ({score:.2f})"
    else:
        text = label

    text_y = max(
        25,
        y - 10
    )

    cv2.putText(
        image,
        text,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# ============================================================
# PROCESS ATTENDANCE IMAGE
# ============================================================

def process_attendance_image(
    image,
    detector,
    recognizer
):
    """
    Detect and recognize all faces in an image.
    """

    students = load_students()
    embeddings = load_embeddings()

    faces = detect_faces(
        image,
        detector
    )

    results = []

    if not faces:
        return image, results

    for face in faces:

        embedding = get_embedding(
            image,
            face,
            recognizer
        )

        if embedding is None:
            draw_face(
                image,
                face,
                "Recognition Error"
            )

            continue

        student_id, score = find_best_match(
            embedding,
            embeddings,
            students
        )

        if student_id is None:

            draw_face(
                image,
                face,
                "Unknown",
                score
            )

            results.append({
                "student_id": None,
                "name": "Unknown",
                "score": score,
                "status": "Unknown"
            })

            continue

        student = students.get(
            student_id,
            {}
        )

        name = student.get(
            "name",
            student_id
        )

        marked, message = mark_attendance(
            student_id,
            name
        )

        if marked:

            status = "Present"

        else:

            status = "Already Marked"

        draw_face(
            image,
            face,
            name,
            score
        )

        results.append({
            "student_id": student_id,
            "name": name,
            "score": score,
            "status": status
        })

    return image, results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 AI Attendance")

    st.caption(
        "Lightweight Face Recognition System"
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👤 Register Student",
            "📷 Mark Attendance",
            "📊 Attendance Records",
            "👥 Students",
        ]
    )

    st.divider()

    students_count = len(
        load_students()
    )

    attendance_count = len(
        load_attendance()
    )

    st.metric(
        "Students",
        students_count
    )

    st.metric(
        "Attendance Entries",
        attendance_count
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    detector, recognizer = load_models()

    models_ready = True

except Exception as e:

    models_ready = False

    st.error(
        f"Model initialization failed:\n\n{e}"
    )

    st.info(
        "Make sure the ONNX models are present in the models folder."
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("📊 AI Attendance System")

    st.write(
        "Lightweight face-recognition based attendance management."
    )

    st.divider()

    students = load_students()
    attendance = load_attendance()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "👥 Total Students",
            len(students)
        )

    with col2:
        st.metric(
            "📅 Total Attendance",
            len(attendance)
        )

    today_string = str(date.today())

    today_attendance = attendance[
        attendance["date"] == today_string
    ]

    with col3:
        st.metric(
            "✅ Today's Attendance",
            len(today_attendance)
        )

    st.divider()

    st.subheader("System Status")

    if models_ready:

        st.success(
            "AI models loaded successfully."
        )

    else:

        st.error(
            "AI models are not ready."
        )

    st.write(
        f"**YuNet:** `{YUNET_MODEL.name}`"
    )

    st.write(
        f"**SFace:** `{SFACE_MODEL.name}`"
    )

    st.write(
        f"**Matching threshold:** `{MATCH_THRESHOLD}`"
    )


# ============================================================
# REGISTER STUDENT
# ============================================================

elif page == "👤 Register Student":

    st.title("👤 Register Student")

    st.write(
        "Register a student using a clear photo containing exactly one face."
    )

    if not models_ready:

        st.warning(
            "AI models are not available."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.text_input(
                "Student ID",
                placeholder="Example: 19"
            )

            student_name = st.text_input(
                "Student Name",
                placeholder="Example: Abhishek Tiwari"
            )

        with col2:

            uploaded_photo = st.file_uploader(
                "Upload Student Photo",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

        if uploaded_photo:

            image = bytes_to_cv2(
                uploaded_photo
            )

            if image is None:

                st.error(
                    "Could not read the image."
                )

            else:

                st.image(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Uploaded Photo",
                    width=400
                )

                faces = detect_faces(
                    image,
                    detector
                )

                st.write(
                    f"Detected faces: **{len(faces)}**"
                )

                if len(faces) == 1:

                    embedding = get_embedding(
                        image,
                        faces[0],
                        recognizer
                    )

                    if embedding is not None:

                        st.success(
                            "Exactly one face detected. Ready for registration."
                        )

                        if st.button(
                            "➕ Register Student",
                            type="primary"
                        ):

                            if not student_id.strip():

                                st.error(
                                    "Please enter Student ID."
                                )

                            elif not student_name.strip():

                                st.error(
                                    "Please enter Student Name."
                                )

                            else:

                                students = load_students()
                                embeddings = load_embeddings()

                                clean_id = safe_id(
                                    student_id
                                )

                                if clean_id in students:

                                    st.error(
                                        "This Student ID is already registered."
                                    )

                                else:

                                    students[clean_id] = {
                                        "name": student_name.strip(),
                                        "registered_at": datetime.now().strftime(
                                            "%Y-%m-%d %H:%M:%S"
                                        )
                                    }

                                    embeddings[
                                        clean_id
                                    ] = embedding

                                    save_students(
                                        students
                                    )

                                    save_embeddings(
                                        embeddings
                                    )

                                    # Save original photo
                                    photo_path = (
                                        PHOTOS_DIR /
                                        f"{clean_id}.jpg"
                                    )

                                    cv2.imwrite(
                                        str(photo_path),
                                        image
                                    )

                                    st.success(
                                        f"Student '{student_name}' registered successfully!"
                                    )

                                    st.rerun()

                    else:

                        st.error(
                            "Could not generate face embedding."
                        )

                elif len(faces) == 0:

                    st.error(
                        "No face detected. Please use a clearer photo."
                    )

                else:

                    st.error(
                        "Multiple faces detected. "
                        "Registration requires exactly one face."
                    )


# ============================================================
# MARK ATTENDANCE
# ============================================================

elif page == "📷 Mark Attendance":

    st.title("📷 Mark Attendance")

    st.write(
        "Use your camera or upload an image containing one or more registered students."
    )

    if not models_ready:

        st.warning(
            "AI models are not available."
        )

    else:

        students = load_students()

        if not students:

            st.info(
                "No students registered yet. "
                "Go to 'Register Student' first."
            )

        else:

            input_method = st.radio(
                "Choose input method",
                [
                    "📷 Camera",
                    "📁 Upload Image"
                ],
                horizontal=True
            )

            uploaded_file = None

            if input_method == "📷 Camera":

                uploaded_file = st.camera_input(
                    "Take attendance photo"
                )

            else:

                uploaded_file = st.file_uploader(
                    "Upload attendance image",
                    type=[
                        "jpg",
                        "jpeg",
                        "png"
                    ]
                )

            if uploaded_file:

                image = bytes_to_cv2(
                    uploaded_file
                )

                if image is None:

                    st.error(
                        "Could not read image."
                    )

                else:

                    processed_image, results = (
                        process_attendance_image(
                            image.copy(),
                            detector,
                            recognizer
                        )
                    )

                    st.image(
                        cv2.cvtColor(
                            processed_image,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Recognition Result",
                        width=800
                    )

                    st.divider()

                    if not results:

                        st.warning(
                            "No faces detected."
                        )

                    else:

                        st.subheader(
                            "Recognition Results"
                        )

                        for result in results:

                            if result["status"] == "Present":

                                st.success(
                                    f"✅ {result['name']} — "
                                    f"Attendance marked "
                                    f"(similarity: {result['score']:.2f})"
                                )

                            elif result["status"] == "Already Marked":

                                st.info(
                                    f"ℹ️ {result['name']} — "
                                    f"Already marked today "
                                    f"(similarity: {result['score']:.2f})"
                                )

                            else:

                                st.warning(
                                    f"❓ Unknown face — "
                                    f"similarity: {result['score']:.2f}"
                                )


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

elif page == "📊 Attendance Records":

    st.title("📊 Attendance Records")

    # ========================================================
    # SAFE CSV LOADING
    # ========================================================

    df = load_attendance()

    # --------------------------------------------------------
    # ALWAYS ensure required columns exist
    # --------------------------------------------------------

    for column in ATTENDANCE_COLUMNS:

        if column not in df.columns:

            df[column] = ""

    df = df[ATTENDANCE_COLUMNS]

    # --------------------------------------------------------
    # Convert values safely
    # --------------------------------------------------------

    df["date"] = (
        df["date"]
        .fillna("")
        .astype(str)
    )

    df["time"] = (
        df["time"]
        .fillna("")
        .astype(str)
    )

    df["student_id"] = (
        df["student_id"]
        .fillna("")
        .astype(str)
    )

    df["name"] = (
        df["name"]
        .fillna("")
        .astype(str)
    )

    # ========================================================
    # FILTERS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        if not df.empty and df["date"].str.len().gt(0).any():

            dates = sorted(
                df["date"]
                .dropna()
                .unique(),
                reverse=True
            )

            selected_date = st.selectbox(
                "Filter by date",
                ["All"] + list(dates)
            )

        else:

            selected_date = "All"

    with col2:

        students = load_students()

        student_options = [
            "All"
        ]

        for sid, info in students.items():

            student_options.append(
                f"{sid} - {info.get('name', sid)}"
            )

        selected_student = st.selectbox(
            "Filter by student",
            student_options
        )

    # ========================================================
    # APPLY FILTERS
    # ========================================================

    filtered_df = df.copy()

    if selected_date != "All":

        filtered_df = filtered_df[
            filtered_df["date"] == selected_date
        ]

    if selected_student != "All":

        selected_id = selected_student.split(
            " - ",
            1
        )[0]

        filtered_df = filtered_df[
            filtered_df["student_id"] == selected_id
        ]

    # ========================================================
    # SUMMARY
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            len(filtered_df)
        )

    with col2:

        unique_students = (
            filtered_df["student_id"]
            .replace("", np.nan)
            .dropna()
            .nunique()
        )

        st.metric(
            "Students Present",
            unique_students
        )

    with col3:

        if selected_date == "All":

            today_count = len(
                df[
                    df["date"] == str(date.today())
                ]
            )

        else:

            today_count = len(
                filtered_df
            )

        st.metric(
            "Selected Day",
            today_count
        )


        st.divider()

st.subheader("⚠️ Data Management")

if st.button("🗑️ Reset All Data", use_container_width=True):
    st.session_state["confirm_reset"] = True

if st.session_state.get("confirm_reset", False):

    st.warning(
        "This will permanently delete all registered students, "
        "face embeddings, student photos, and attendance records."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes, Reset", type="primary"):
            # Reset students
            with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4)

            # Reset embeddings
            np.savez_compressed(EMBEDDINGS_FILE)

            # Reset attendance
            pd.DataFrame(
                columns=ATTENDANCE_COLUMNS
            ).to_csv(
                ATTENDANCE_FILE,
                index=False
            )

            # Delete student photos
            for photo in PHOTOS_DIR.iterdir():
                if photo.is_file():
                    try:
                        photo.unlink()
                    except Exception:
                        pass

            st.session_state["confirm_reset"] = False

            st.success("✅ All attendance data has been reset.")
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.session_state["confirm_reset"] = False
            st.rerun()

    st.divider()

    # ========================================================
    # DISPLAY
    # ========================================================

    if filtered_df.empty:

        st.info(
            "No attendance records found."
        )

    else:

        display_df = filtered_df.sort_values(
            by=["date", "time"],
            ascending=False
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv_data = display_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Attendance CSV",
            data=csv_data,
            file_name="attendance_records.csv",
            mime="text/csv"
        )


# ============================================================
# STUDENTS
# ============================================================

elif page == "👥 Students":

    st.title("👥 Registered Students")

    students = load_students()

    embeddings = load_embeddings()

    if not students:

        st.info(
            "No students registered yet."
        )

    else:

        st.write(
            f"Total registered students: **{len(students)}**"
        )

        st.divider()

        for student_id, student in students.items():

            name = student.get(
                "name",
                student_id
            )

            registered_at = student.get(
                "registered_at",
                "Unknown"
            )

            with st.container(border=True):

                col1, col2, col3 = st.columns(
                    [3, 3, 1]
                )

                with col1:

                    st.subheader(
                        name
                    )

                    st.write(
                        f"**ID:** {student_id}"
                    )

                with col2:

                    st.write(
                        f"Registered: {registered_at}"
                    )

                    if student_id in embeddings:

                        st.success(
                            "Embedding available"
                        )

                    else:

                        st.warning(
                            "Embedding missing"
                        )

                with col3:

                    delete_key = (
                        f"delete_{student_id}"
                    )

                    if st.button(
                        "🗑️ Delete",
                        key=delete_key
                    ):

                        st.session_state[
                            "delete_student"
                        ] = student_id

                # --------------------------------------------
                # Delete confirmation
                # --------------------------------------------

                if st.session_state.get(
                    "delete_student"
                ) == student_id:

                    st.warning(
                        f"Delete {name} ({student_id})?"
                    )

                    confirm_col1, confirm_col2 = (
                        st.columns(2)
                    )

                    with confirm_col1:

                        if st.button(
                            "Yes, Delete",
                            key=f"confirm_{student_id}"
                        ):

                            students = load_students()

                            embeddings = load_embeddings()

                            students.pop(
                                student_id,
                                None
                            )

                            embeddings.pop(
                                student_id,
                                None
                            )

                            save_students(
                                students
                            )

                            save_embeddings(
                                embeddings
                            )

                            # Delete stored photo
                            photo_path = (
                                PHOTOS_DIR /
                                f"{safe_id(student_id)}.jpg"
                            )

                            if photo_path.exists():

                                try:
                                    photo_path.unlink()

                                except Exception:
                                    pass

                            st.session_state.pop(
                                "delete_student",
                                None
                            )

                            st.success(
                                "Student deleted."
                            )

                            st.rerun()

                    with confirm_col2:

                        if st.button(
                            "Cancel",
                            key=f"cancel_{student_id}"
                        ):

                            st.session_state.pop(
                                "delete_student",
                                None
                            )

                            st.rerun()