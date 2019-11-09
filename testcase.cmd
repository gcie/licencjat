@echo off

if not "%2"=="sync" (
    mkdir "tests/%1"
    cp -r src "tests/%1"

    cp config.test.py "tests/%1/config.py"
    cp main.py "tests/%1/main.py"
    set /a "r1=%RANDOM%*32768+%RANDOM%"
    set /a "r2=%RANDOM%*32768+%RANDOM%"
    powershell -Command "(gc tests/%1/main.py) -replace '12345678', '%r1%' | Out-File -encoding ASCII tests/%1/main.py"
    powershell -Command "(gc tests/%1/main.py) -replace '87654321', '%r2%' | Out-File -encoding ASCII tests/%1/main.py"
    powershell -Command "(gc tests/%1/main.py) -replace 'main_test', '%1' | Out-File -encoding ASCII tests/%1/main.py"
)


if "%2"=="remote" (
    powershell -Command "(gc tests/%1/src/remote.py) -replace 'remote = False', 'remote = True' | Out-File -encoding ASCII tests/%1/src/remote.py"
    set HOST="grzesiek@guccihome.ddns.net"
    ssh "%HOST%" "mkdir ~/licencjat/%1"
    scp "tests/%1/main.py" "%HOST%:~/licencjat/%1/main.py"
    scp "tests/%1/config.py" "%HOST%:~/licencjat/%1/config.py"
    scp -r "tests/%1/src" "%HOST%:~/licencjat/%1/src"
)
if "%2"=="sync" (
    set HOST="grzesiek@guccihome.ddns.net"
    scp "%HOST%:~/licencjat/%1/main.py" "tests/%1/main.py"
    scp "%HOST%:~/licencjat/%1/config.py" "tests/%1/config.py"
    scp -r "%HOST%:~/licencjat/%1/src" "tests/%1"
)