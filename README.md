# 📄 Smart ATS Resume Analyzer

A Streamlit web app that compares your resume with a job description and tells you how well you match, which keywords are missing, and exactly what to improve. It is powered by Google Gemini and combines AI analysis with rule-based ATS format checks.

## Features

- **Match score (0-100%)** based on a weighted rubric: skills, relevant experience, role fit, education and impact.
- **Matched and missing keywords** between your resume and the job description.
- **Strengths and specific improvements** that tell you what to change and where.
- **ATS format check** (rule-based, not AI): contact details, standard section headings, resume length and use of metrics.
- **Two input modes:** upload a PDF or paste text (useful for scanned PDFs).
- **Downloadable report** in Markdown.

## How it works

1. The resume text is extracted from the PDF with `pypdf`.
2. The text and job description are placed into a structured prompt, and Gemini replies in a fixed JSON schema.
3. The response is validated: the score is clamped to 0-100, duplicates are removed, and any keyword reported as "missing" that actually appears in the resume is filtered out.
4. Separate rule-based checks look at the resume format, since that is what an ATS parser cares about.

## Setup

```bash
# 1. Go into the project folder
cd resume-analyzer

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env        # Windows: copy .env.example .env
# Open .env and paste your key

# 5. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`. Get a free API key at https://aistudio.google.com/apikey.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface: inputs, results, report download |
| `helper.py` | PDF extraction, ATS format checks, Gemini call, output cleanup |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your API key |
| `.gitignore` | Keeps secrets and junk files out of Git |

## Configuration

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Your Gemini API key |
| `GEMINI_MODEL` | No | Model name. Default: `gemini-2.5-flash` |

## Deploy on Streamlit Community Cloud (free)

1. Push this project to a public GitHub repository (without `.env`).
2. Create a new app on https://share.streamlit.io and select `app.py`.
3. In **App settings -> Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```

## Troubleshooting

- **"API key not found"**: the `.env` file must be in the project root, next to `app.py`.
- **"No text could be extracted"**: the PDF is probably a scanned image. Export a text-based PDF from Word or Google Docs, or use the "Paste text" option.
- **"Rate limit or quota reached"**: wait a minute and retry; the free tier has limits.
- **"Model was not found"**: set `GEMINI_MODEL` in `.env` to a currently available model.

## Limitations

- The match score is an AI estimate, not the result of any real company's ATS. Use it as guidance.
- Only text-based PDFs are supported for upload.
- Resume text is sent to the Gemini API for analysis. The app itself does not store it.

## Security

Never commit your `.env` file. If a key is ever exposed, revoke it immediately in Google AI Studio and create a new one.

## Tech stack

Python, Streamlit, Google Gemini (`google-genai`), pypdf, python-dotenv
