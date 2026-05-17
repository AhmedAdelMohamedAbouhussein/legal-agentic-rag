@echo off
title Legal AI Assistant - Agentic RAG
echo.
echo  ============================================
echo    Legal AI Assistant - Agentic RAG
echo    Ahmed Adel Abouhussein - 231007728
echo  ============================================
echo.

:: Activate virtual environment
call .venv\Scripts\activate

:: Run the Streamlit app
echo  Starting Streamlit server...
echo  Open http://localhost:8501 in your browser
echo.
streamlit run app.py

pause
