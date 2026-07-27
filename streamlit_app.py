from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

from PIL import Image
import streamlit as st

from backend.inference import run_pipeline
from backend.model_loader import load_models
from backend.text_to_speech import synthesize_speech
from backend.translator import translate_braille


st.set_page_config(
    page_title="Braille Reader",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("Braille Reader")

with st.sidebar:
    st.subheader("Detection settings")
    detector_confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    translation_grade = st.selectbox(
        "Braille translation",
        options=[1, 2],
        format_func=lambda grade: (
            "Grade 1 (uncontracted)"
            if grade == 1
            else "Grade 2 (contracted)"
        ),
    )

source_mode = st.segmented_control(
    "Image source",
    options=["Upload", "Camera"],
    default="Upload",
)

if source_mode == "Camera":
    source = st.camera_input("Take a Braille photo")
else:
    source = st.file_uploader(
        "Upload a Braille image",
        type=["jpg", "jpeg", "png", "webp"],
    )

image_bytes: bytes | None = None
image: Image.Image | None = None
source_signature: str | None = None

if source is not None:
    image_bytes = source.getvalue()
    source_signature = sha256(image_bytes).hexdigest()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Input image", width="stretch")

if st.session_state.get("source_signature") != source_signature:
    st.session_state["source_signature"] = source_signature
    st.session_state.pop("pipeline_result", None)
    st.session_state.pop("audio_result", None)
    st.session_state.pop("audio_requested", None)

analyze_clicked = st.button(
    "Read Braille",
    type="primary",
    disabled=image is None,
)

if analyze_clicked and image is not None:
    status = st.status("Recognizing Braille...", expanded=True)
    try:
        status.write("Loading YOLO11n and EfficientNet-B0")
        model_bundle = get_models()
        status.write("Detecting and classifying Braille cells")
        pipeline_result = run_pipeline(
            image=image,
            models=model_bundle,
            detector_confidence=detector_confidence,
        )

        st.session_state["pipeline_result"] = pipeline_result
        st.session_state.pop("audio_result", None)
        st.session_state.pop("audio_requested", None)
        status.update(
            label="Braille recognition complete",
            state="complete",
            expanded=False,
        )
    except Exception as exc:
        status.update(
            label="Braille recognition failed",
            state="error",
            expanded=True,
        )
        st.exception(exc)

result = st.session_state.get("pipeline_result")

if result is not None:
    translation_result = translate_braille(
        result["recognized_text"],
        grade=translation_grade,
    )

    st.subheader("Recognition result")
    st.image(
        result["annotated_image"],
        caption="Detected and classified Braille cells",
        width="stretch",
    )

    if not result["predictions"]:
        st.warning("No Braille characters were detected.")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Detected characters", len(result["predictions"]))
    metric_columns[1].metric(
        "Inference time",
        f"{result['elapsed_seconds']:.2f} s",
    )
    metric_columns[2].metric("Device", result["device"])

    if not translation_result.available:
        st.warning(translation_result.error)

    edited_text = st.text_area(
        "Recognized English text",
        value=translation_result.text,
        height=140,
        key=f"recognized_text_{source_signature}_{translation_grade}",
    )

    with st.expander("Braille translation details"):
        detail_columns = st.columns(2)
        detail_columns[0].text_area(
            "Raw cell labels",
            value=translation_result.raw_labels,
            height=120,
            disabled=True,
        )
        detail_columns[1].text_area(
            "Unicode Braille",
            value=translation_result.braille_cells,
            height=120,
            disabled=True,
        )

    action_columns = st.columns(3)
    action_columns[0].download_button(
        "Download text",
        data=edited_text,
        file_name="recognized_braille.txt",
        mime="text/plain",
        disabled=not edited_text,
        width="stretch",
    )
    action_columns[1].download_button(
        "Download image",
        data=image_to_png_bytes(result["annotated_image"]),
        file_name="braille_detection.png",
        mime="image/png",
        width="stretch",
    )

    if action_columns[2].button(
        "Create audio",
        disabled=not edited_text,
        width="stretch",
    ):
        st.session_state["audio_result"] = synthesize_speech(edited_text)
        st.session_state["audio_requested"] = True

    audio_result = st.session_state.get("audio_result")
    if audio_result is not None:
        st.audio(audio_result.data, format=audio_result.mime_type)
    elif st.session_state.get("audio_requested"):
        st.info("Text-to-speech integration is pending.")

    with st.expander("Character predictions"):
        table_rows = [
            {
                "Order": prediction["reading_index"],
                "Line": prediction["line"],
                "Letter": prediction["letter"],
                "Detector confidence": prediction["detector_confidence"],
                "Classifier confidence": prediction["classifier_confidence"],
                "Box": prediction["box"],
            }
            for prediction in result["predictions"]
        ]
        st.dataframe(table_rows, hide_index=True, width="stretch")
