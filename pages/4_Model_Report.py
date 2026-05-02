"""
pages/4_Model_Report.py — Performance report
"""

import pandas as pd
import streamlit as st

from shared import (
    REPORT_AVAILABLE,
    cached_report,
    inject_styles,
)

inject_styles()

st.title("📊 Model Performance Report")
st.caption("Evaluation on the held-out 20% test split of your dataset.")

if not REPORT_AVAILABLE:
    st.error(
        "`model_report.py` not found. "
        "Make sure it is in the project root alongside `streamlit_app.py`."
    )
    st.stop()

# ── Refresh button ────────────────────────────────────────────────────────────
if st.button("🔄  Re-run evaluation", help="Clears cache and re-evaluates the model"):
    cached_report.clear()
    st.rerun()

# ── Generate report ───────────────────────────────────────────────────────────
with st.spinner("Evaluating model on test data..."):
    report = cached_report()

if report.error:
    st.error(report.error)
    st.stop()


# ── 1. Overall metric cards ───────────────────────────────────────────────────
st.subheader("Overall metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Test accuracy",  f"{report.test_accuracy * 100:.2f}%")
c2.metric("Test loss",      f"{report.test_loss:.4f}")
c3.metric("Letters",        str(report.num_classes))
c4.metric("Model params",   f"{report.total_params:,}")
c5.metric("Train samples",  f"{report.train_samples:,}")
c6.metric("Test samples",   f"{report.test_samples:,}")

st.divider()


# ── 2. Per-letter metrics ─────────────────────────────────────────────────────
def _f1_color(val: float) -> str:
    if val >= 0.90:
        return "background-color: #d4edda; color: #1a5c2e"
    if val >= 0.75:
        return "background-color: #fff3cd; color: #7a5200"
    return "background-color: #f8d7da; color: #7a1a1a"


st.subheader("Per-letter metrics")
st.caption("🟢 F1 ≥ 0.90   🟡 F1 ≥ 0.75   🔴 F1 < 0.75")

if not report.per_letter_df.empty:
    tab_table, tab_f1, tab_pr = st.tabs(
        ["Table", "F1 score chart", "Precision vs Recall"]
    )

    with tab_table:
        # pandas >= 2.1 renamed applymap -> map; support both versions
        _styler = (
            report.per_letter_df.style
            .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1 score": "{:.3f}"})
            .set_properties(**{"text-align": "center"})
        )
        try:
            _styler = _styler.map(_f1_color, subset=["F1 score"])
        except AttributeError:
            _styler = _styler.applymap(_f1_color, subset=["F1 score"])
        st.dataframe(_styler, use_container_width=True, hide_index=True)

    with tab_f1:
        st.bar_chart(
            report.per_letter_df.set_index("Letter")[["F1 score"]],
            height=320,
        )

    with tab_pr:
        st.bar_chart(
            report.per_letter_df.set_index("Letter")[["Precision", "Recall"]],
            height=320,
        )
else:
    st.info("No per-letter data available.")

st.divider()


# ── 3. Confusion matrix ───────────────────────────────────────────────────────
st.subheader("Confusion matrix")
st.caption(
    "Rows = true letter · Columns = predicted letter. "
    "Green diagonal = correct predictions. Red off-diagonal = errors."
)

if not report.confusion_df.empty:
    cm_vals = report.confusion_df.values.astype(float)
    max_val = cm_vals.max() if cm_vals.max() > 0 else 1

    def highlight_cm(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                v = data.iloc[i, j]
                if i == j:
                    g = int(80 + 160 * v / max_val)
                    styles.iloc[i, j] = (
                        f"background-color: rgb(30,{g},60); "
                        "color: white; font-weight: 600"
                    )
                elif v > 0:
                    r = int(220 - 140 * v / max_val)
                    styles.iloc[i, j] = (
                        f"background-color: rgb({r},50,50); color: white"
                    )
        return styles

    st.dataframe(
        report.confusion_df.style.apply(highlight_cm, axis=None),
        use_container_width=True,
    )
else:
    st.info("No confusion matrix data available.")

st.divider()


# ── 4. Training history image ─────────────────────────────────────────────────
if report.history_img_path:
    st.subheader("Training history")
    st.caption("Accuracy and loss curves from the last training run.")
    st.image(report.history_img_path, use_container_width=True)
    st.divider()


# ── 5. Dataset statistics ─────────────────────────────────────────────────────
st.subheader("Dataset statistics")
st.caption("Sample count per letter in data/dataset.csv")

if not report.dataset_df.empty:
    total = int(report.dataset_df["Samples"].sum())
    avg   = float(report.dataset_df["Samples"].mean())
    mn    = int(report.dataset_df["Samples"].min())
    mx    = int(report.dataset_df["Samples"].max())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total samples",    f"{total:,}")
    d2.metric("Average / letter", f"{avg:.0f}")
    d3.metric("Min / letter",     str(mn))
    d4.metric("Max / letter",     str(mx))

    st.bar_chart(
        report.dataset_df.set_index("Letter")[["Samples"]],
        height=280,
    )

    if mn < 100:
        weak = report.dataset_df[
            report.dataset_df["Samples"] < 100
        ]["Letter"].tolist()
        st.warning(
            f"Letters with fewer than 100 samples: **{', '.join(weak)}**  \n"
            "Collecting more data for these will improve accuracy."
        )
else:
    st.info("No dataset statistics available.")