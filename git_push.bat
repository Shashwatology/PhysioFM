@echo off
echo Initializing Git Repository...
git init

echo Adding GitHub Remote...
git remote add origin https://github.com/Shashwatology/contactless-respiratory-monitoring

echo Staging files...
git add .

echo Committing...
git commit -m "Initial research infrastructure for PhysioFM: reproducible training, evaluation pipeline, documentation, and cloud-ready execution."

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause
