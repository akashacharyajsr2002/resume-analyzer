"""
app.py - Streamlit interface for the Smart ATS Resume Analyzer.

Run with:  streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from helper import (
    AnalysisError,
    analyze,
    ats_format_checks,
    build_report,
    clean_text,
    extract_pdf_text,
    get_client,
    get_model_name,
    score_label,
)

load_dotenv()

MAX_PDF_MB = 5

st.set_page_config(
    page_title="Smart ATS Resume Analyzer",
    page_icon="📄",
    layout="wide",
)


def get_api_key() -> str:
    """Read the key from Streamlit secrets first (deployment), then from .env (local)."""
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY", "")


@st.cache_resource(show_spinner=False)
def cached_client(api_key: str):
    return get_client(api_key)


def render_sidebar(api_key: str) -> None:
    with st.sidebar:
        st.title("🎯 Smart ATS")
        st.write(
            "Compare your resume with a job description and get:\n\n"
            "- A match score\n"
            "- Matched and missing keywords\n"
            "- Specific improvement suggestions\n"
            "- An ATS format check"
        )
        st.divider()
        if api_key:
            st.success("API key detected")
        else:
            st.error("API key not found")
        st.caption(f"Model: `{get_model_name()}`")
        st.caption(
            "Privacy: your resume text is sent to Google's Gemini API for analysis. "
            "This app does not store it."
        )


def render_results(data: dict, checks: list[dict]) -> None:
    score = data["jd_match"]
    st.divider()

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Match score", f"{score}%")
        st.progress(score / 100)
        st.caption(
            f"**{score_label(score)}** - 75%+ strong, 50-74% moderate, below 50% weak."
        )
    with right:
        st.subheader("Summary")
        st.write(data["profile_summary"] or "No summary was returned.")

    tab_keywords, tab_advice, tab_format = st.tabs(
        ["Keywords", "Strengths and improvements", "ATS format check"]
    )

    with tab_keywords:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Matched keywords")
            if data["matched_keywords"]:
                for kw in data["matched_keywords"]:
                    st.markdown(f"- {kw}")
            else:
                st.write("No matches were identified.")
        with c2:
            st.subheader("Missing keywords")
            if data["missing_keywords"]:
                for kw in data["missing_keywords"]:
                    st.markdown(f"- {kw}")
            else:
                st.write("No important keywords are missing.")

    with tab_advice:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Strengths")
            for item in data["strengths"] or ["-"]:
                st.markdown(f"- {item}")
        with c2:
            st.subheader("Improvements")
            if data["improvements"]:
                for i, item in enumerate(data["improvements"], 1):
                    st.markdown(f"**{i}.** {item}")
            else:
                st.write("-")

    with tab_format:
        passed = sum(c["passed"] for c in checks)
        st.write(f"**{passed} of {len(checks)} checks passed.** These are rule-based checks, not AI.")
        for c in checks:
            icon = "✅" if c["passed"] else "⚠️"
            st.markdown(f"{icon} **{c['name']}** - {c['detail']}")

    st.download_button(
        "Download report (.md)",
        data=build_report(data, checks),
        file_name="resume_analysis_report.md",
        mime="text/markdown",
    )


def main() -> None:
    api_key = get_api_key()
    render_sidebar(api_key)

    st.title("📄 Smart ATS Resume Analyzer")
    st.caption("Optimise your resume for a specific job. Powered by Google Gemini.")

    if not api_key:
        st.warning(
            "Add your API key before using the app. Create a `.env` file in the project "
            "folder containing:\n\n```\nGOOGLE_API_KEY=your_key_here\n```\n\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
        return

    col_jd, col_resume = st.columns(2, gap="large")

    with col_jd:
        job_description = st.text_area(
            "Job description",
            height=300,
            placeholder="Paste the full job description here...",
        )

    with col_resume:
        mode = st.radio("Resume input", ["Upload PDF", "Paste text"], horizontal=True)
        uploaded_file = None
        pasted_text = ""
        if mode == "Upload PDF":
            uploaded_file = st.file_uploader("Resume (PDF)", type="pdf")
            st.caption("Use a text-based PDF (not a scanned image).")
        else:
            pasted_text = st.text_area("Resume text", height=240, placeholder="Paste your resume text here...")

    if st.button("Analyze resume", type="primary", use_container_width=True):
        if not job_description.strip():
            st.warning("Please paste the job description first.")
            return

        try:
            with st.spinner("Analyzing your resume..."):
                if mode == "Upload PDF":
                    if uploaded_file is None:
                        st.warning("Please upload your resume as a PDF.")
                        return
                    if uploaded_file.size > MAX_PDF_MB * 1024 * 1024:
                        st.error(f"The file is larger than {MAX_PDF_MB} MB.")
                        return
                    resume_text = extract_pdf_text(uploaded_file)
                else:
                    resume_text = clean_text(pasted_text)
                    if not resume_text:
                        st.warning("Please paste your resume text.")
                        return

                data = analyze(cached_client(api_key), resume_text, job_description)
                checks = ats_format_checks(resume_text)
        except (ValueError, AnalysisError) as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # last-resort guard so the app never shows a traceback
            st.error(f"Something went wrong: {exc}")
            return

        st.session_state["result"] = {"data": data, "checks": checks}

    result = st.session_state.get("result")
    if result:
        render_results(result["data"], result["checks"])


if __name__ == "__main__":
    main()
