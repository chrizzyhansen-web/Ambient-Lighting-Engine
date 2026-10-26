@echo off
title LAN control test
echo Checking whether your strip can be reached over the local network.
echo.
echo This is the fast path. If it works, gradients are drawn per segment
echo about 25 times a second. If it doesn't, everything has to go through
echo Govee's cloud API, which only allows 10 changes per minute.
echo.
echo Nothing here changes your settings. It will briefly change the
echo colour of your strip if it finds it.
echo.
py -3 test_lan.py
echo.
pause
