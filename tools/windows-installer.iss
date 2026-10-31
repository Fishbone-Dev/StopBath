; Optional Windows installer for Stop Bath (Inno Setup 6: https://jrsoftware.org/isinfo.php).
; Build "Stop Bath.exe" first (tools/build-windows.sh), then compile this file with ISCC.exe.
[Setup]
AppName=Stop Bath
AppVersion=1.1.0
AppPublisher=Fishbone
AppCopyright=© 2026 Fishbone
DefaultDirName={autopf}\Stop Bath
DefaultGroupName=Stop Bath
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=StopBath-1.1.0-Windows-x64-Setup
SetupIconFile=..\stop-bath.ico
UninstallDisplayIcon={app}\Stop Bath.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\build-win\dist\Stop Bath.exe"; DestDir: "{app}"
Source: "..\stopbath\luts\CREDITS.txt"; DestDir: "{app}"; DestName: "Credits.txt"

[Icons]
Name: "{group}\Stop Bath"; Filename: "{app}\Stop Bath.exe"
Name: "{autodesktop}\Stop Bath"; Filename: "{app}\Stop Bath.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Stop Bath.exe"; Description: "Start Stop Bath"; Flags: nowait postinstall skipifsilent
