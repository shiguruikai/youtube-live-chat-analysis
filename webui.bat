@echo off
cd /d %~dp0

set PYTHON=python
set "VENV_DIR=%~dp0%venv"

%PYTHON% --version >nul 2>&1
if %ERRORLEVEL% == 0 goto :check_venv
echo pythonを起動できませんでした。
goto :abend

:check_venv
dir "%VENV_DIR%\Scripts\python.exe" >nul 2>&1
if %ERRORLEVEL% == 0 goto :activate_venv
goto :create_venv

:create_venv
echo pythonで仮想環境の %VENV_DIR% ディレクトリを作成します。
%PYTHON% -m venv %VENV_DIR%
if %ERRORLEVEL% == 0 goto :activate_venv
echo 仮想環境 %VENV_DIR% を作成できませんでした。
goto :abend

:activate_venv
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
goto :check_streamlit

:check_streamlit
%PYTHON% -m streamlit --version >nul 2>&1
if %ERRORLEVEL% == 0 goto :launch
goto :install

:install
echo パッケージをインストールします。
%PYTHON% -m pip install -U -r requirements.txt
if %ERRORLEVEL% == 0 goto :launch
echo パッケージのインストールに失敗しました。
goto :abend

:launch
%PYTHON% -m streamlit run app.py
pause
exit /b

:abend
pause
exit /b 1
