@echo off

cp -r src "tests/%1"
mv "tests/%1/main.py" "tests/%1/%1.py"
cd tests/%1
