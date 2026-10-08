

import streamlit as st

from predict import suggest_next_words


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NextWord AI",
    page_icon="✍️",
    layout="centered"
)


# --------------------------------------------------
# 2. Page styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .suggestion-card {
        padding: 14px;
        border: 1px solid #dddddd;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 3. Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">NextWord AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict your next word using a Transformer model'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 4. Session state
# --------------------------------------------------

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None


def add_suggested_word(word):
    """Append a selected suggestion to the current sentence."""
    current = st.session_state.sentence.strip()

    if current:
        st.session_state.sentence = current + " " + word
    else:
        st.session_state.sentence = word

    # Clear old predictions after changing the input.
    st.session_state.suggestions = []
    st.session_state.prediction_error = None


# --------------------------------------------------
# 5. Sentence input
# --------------------------------------------------

st.subheader("Write your sentence")

st.text_area(
    "Enter some text, then ask the model to suggest the next words.",
    key="sentence",
    height=120,
    placeholder="Example: I am learning"
)


# --------------------------------------------------
# 6. Model settings
# --------------------------------------------------

st.subheader("Model settings")

use_residual = st.toggle(
    "Enable residual connections",
    value=True,
    help=(
        "Turn this on or off to compare the two trained "
        "model variants, if both checkpoints are available."
    )
)

st.caption(
    "ON uses the residual-enabled checkpoint; OFF uses the "
    "residual-disabled checkpoint."
)


# --------------------------------------------------
# 7. Generate predictions
# --------------------------------------------------

if st.button("Suggest next words", type="primary", use_container_width=True):

    if not st.session_state.sentence.strip():
        st.session_state.suggestions = []
        st.session_state.prediction_error = None
        st.warning("Please enter a sentence first.")

    else:
        try:
            with st.spinner("Predicting the next words..."):

                results = suggest_next_words(
                    text=st.session_state.sentence,
                    top_k=3,
                    use_residual=use_residual
                )

            st.session_state.suggestions = results
            st.session_state.prediction_error = None

        except FileNotFoundError as error:
            st.session_state.suggestions = []
            st.session_state.prediction_error = str(error)

        except (ValueError, TypeError, RuntimeError) as error:
            st.session_state.suggestions = []
            st.session_state.prediction_error = str(error)


# --------------------------------------------------
# 8. Display errors
# --------------------------------------------------

if st.session_state.prediction_error:
    st.error(
        "Prediction could not be completed. Check that your "
        "vocabulary and trained model checkpoint exist, and that "
        "the model configuration matches training."
    )
    st.code(st.session_state.prediction_error)


# --------------------------------------------------
# 9. Display suggestions
# --------------------------------------------------

if st.session_state.suggestions:

    st.divider()
    st.subheader("Next-word suggestions")

    st.write(
        "Choose a word below to add it to your sentence."
    )

    for index, item in enumerate(st.session_state.suggestions):

        word = item["word"]
        probability = item["probability"]

        with st.container(border=True):

            left, right = st.columns([3, 1])

            with left:
                st.markdown(f"### {index + 1}. {word}")
                st.progress(
                    max(0.0, min(1.0, probability)),
                    text=f"Model probability: {probability:.1%}"
                )

            with right:
                st.button(
                    "Use word",
                    key=f"use_word_{index}",
                    on_click=add_suggested_word,
                    args=(word,),
                    use_container_width=True
                )


# --------------------------------------------------
# 10. Additional information
# --------------------------------------------------

st.divider()

with st.expander("About this project"):
    st.write(
        """
        NextWord AI is a small next-word prediction application
        built using a Transformer neural network.

        The project demonstrates:
        - Text preprocessing and vocabulary encoding.
        - Transformer-based next-word prediction.
        - Residual connections enabled and disabled.
        - Interactive prediction through Streamlit.
        """
    )

st.caption("NextWord AI | Transformer and Residual Connections Demo")
