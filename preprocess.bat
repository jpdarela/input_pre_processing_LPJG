@echo off

set /P region="Which region to extract? One of { sa af as eu global} (Default sa): "

@echo  Extraction for %region%
py preprocess_lpjginput.py %region%

pause
