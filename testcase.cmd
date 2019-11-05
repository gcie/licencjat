@echo off

mkdir "tests/%1"
cp -r src "tests/%1"



cp config.test.py "tests/%1/config.py"
cp main.py "tests/%1/main.py"
powershell -Command "(gc tests/%1/main.py) -replace '1234567890', '%RANDOM%' | Out-File -encoding ASCII tests/%1/main.py"
powershell -Command "(gc tests/%1/main.py) -replace '9876543210', '%RANDOM%' | Out-File -encoding ASCII tests/%1/main.py"
powershell -Command "(gc tests/%1/main.py) -replace 'main_test', '%1' | Out-File -encoding ASCII tests/%1/main.py"
cd "tests/%1"
code .
cd ..\..
