@echo off
REM Launch the Transformer chatbot (GPU only)
cd /d "%~dp0"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || (
  echo.
  echo  [ERROR] No CUDA GPU detected. This project runs exclusively on the GPU.
  echo          Install a CUDA build of PyTorch:
  echo          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  echo.
  pause
  exit /b 1
)
streamlit run app.py
