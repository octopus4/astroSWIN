if exist "%LocalAppData%\aswin\" rd /q /s "%LocalAppData%\aswin"
mkdir %LocalAppData%\aswin
xcopy /s/y AppData %LocalAppData%\aswin
