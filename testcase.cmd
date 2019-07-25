@echo off

mkdir "tests/%1"
cp -r src "tests/%1"
cp main.py "tests/%1/main.py"
cp config.test.py "tests/%1/config.py"
cd tests/%1
