Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
' Folder containing this VBS:
root = fso.GetParentFolderName(WScript.ScriptFullName)
' Run the BAT hidden (0), don’t wait (False)
shell.Run """" & root & "\Converter_app\Run-App.bat" & """", 0, False
