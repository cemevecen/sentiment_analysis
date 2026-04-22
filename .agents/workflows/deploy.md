---
description: Changes are staged, committed, and pushed to GitHub. This also updates the live app (hosted at Streamlit Cloud) as it reflects the GitHub state.
---
# Deployment Workflow

1. Stage all relevant changes (excluding secrets):
   `git add streamlit_app.py test_gemini.py requirements.txt`
   *(Optional: add other relevant files as needed)*

2. Create a commit with a descriptive message:
// turbo
   `git commit -m "Update: Gemini API configuration and model fallback improvements"`

3. Push to GitHub:
// turbo
   `git push origin main`

4. Final Check:
   The application on Streamlit Cloud will automatically rebuild from the pushed code.
