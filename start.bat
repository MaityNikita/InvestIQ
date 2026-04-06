@echo off
title InvestIQ — Starting Server...
color 0A

echo.
echo  ====================================================
echo    InvestIQ ^⚡ — AI Financial Dashboard
echo  ====================================================
echo.
echo  [*] Starting Flask server on http://localhost:5000 ...
echo.

:: Start the Flask server in the background
start /B python app.py

:: Wait 2 seconds for Flask to boot up
timeout /t 2 /nobreak >nul

:: Open the browser automatically
echo  [*] Opening browser...
start http://localhost:5000

echo.
echo  [✓] Server is running! 
echo  [✓] Browser opened at http://localhost:5000
echo.
echo  Keep this window open while using InvestIQ.
echo  Press Ctrl+C or close this window to stop the server.
echo.

:: Keep window alive to hold the server process
pause >nul
