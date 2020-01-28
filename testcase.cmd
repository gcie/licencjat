@echo off

if "%3"=="" (
    set HOST="grzesiek@guccihome.ddns.net"
) else (
    set HOST="i290956@%3.stud.ii"
)

if "%2"=="create" (
    mkdir "tests/%1"
    cp -r src "tests/%1"

    cp config.test.py "tests/%1/config.py"
    cp main.py "tests/%1/main.py"
    cp run.sh "tests/%1/run.sh"
    set /a "r1=%RANDOM%*32768+%RANDOM%"
    set /a "r2=%RANDOM%*32768+%RANDOM%"
    powershell -Command "(gc tests/%1/main.py) -replace '12345678', '%r1%' | Out-File -encoding ASCII tests/%1/main.py"
    powershell -Command "(gc tests/%1/main.py) -replace '87654321', '%r2%' | Out-File -encoding ASCII tests/%1/main.py"
    powershell -Command "(gc tests/%1/main.py) -replace 'main_test', '%1' | Out-File -encoding ASCII tests/%1/main.py"
)

if "%2"=="remote" (
    powershell -Command "(gc tests/%1/src/remote.py) -replace 'remote = False', 'remote = True' | Out-File -encoding ASCII tests/%1/src/remote.py"
    ssh "%HOST%" "mkdir ~/licencjat/%1"
    scp -r "tests/%1" "%HOST%:~/licencjat"
)

if "%2"=="sync" (
    scp -r "%HOST%:~/licencjat/%1" "tests"
)
