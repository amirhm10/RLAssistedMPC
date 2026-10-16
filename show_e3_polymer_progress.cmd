@echo off
title E3 Polymer - existing recovery progress
echo This window follows the existing E3 polymer recovery log.
echo It does not start or restart a simulation. Closing it leaves the run working.
powershell.exe -NoLogo -NoProfile -Command "Get-Content -LiteralPath '%~dp0Results\e3\20261015_042751_513bff\polymer_combined_s7\a2\worker.log' -Tail 8 -Wait"
