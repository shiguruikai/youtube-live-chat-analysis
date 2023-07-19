@echo off
cd /d %~dp0

set PYTHON=python
set "VENV_DIR=%~dp0%venv"

%PYTHON% --version >nul 2>&1
if %ERRORLEVEL% == 0 goto :check_venv
echo python���N���ł��܂���ł����B
goto :abend

:check_venv
dir "%VENV_DIR%\Scripts\python.exe" >nul 2>&1
if %ERRORLEVEL% == 0 goto :activate_venv
goto :create_venv

:create_venv
echo python�ŉ��z���� %VENV_DIR% �f�B���N�g�����쐬���܂��B
%PYTHON% -m venv %VENV_DIR%
if %ERRORLEVEL% == 0 goto :activate_venv
echo ���z�� %VENV_DIR% ���쐬�ł��܂���ł����B
goto :abend

:activate_venv
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
goto :check_streamlit

:check_streamlit
%PYTHON% -m streamlit --version >nul 2>&1
if %ERRORLEVEL% == 0 goto :launch
goto :install

:install
echo �p�b�P�[�W���C���X�g�[�����܂��B
%PYTHON% -m pip install -U -r requirements.txt
if %ERRORLEVEL% == 0 goto :launch
echo �p�b�P�[�W�̃C���X�g�[���Ɏ��s���܂����B
goto :abend

:launch
%PYTHON% -m streamlit run app.py
pause
exit /b

:abend
pause
exit /b 1
