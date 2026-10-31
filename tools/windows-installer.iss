; Optional Windows installer for Film Lab (Inno Setup 6: https://jrsoftware.org/isinfo.php).
; Build "Film Lab.exe" first (tools/build-windows.sh), then compile this file with ISCC.exe.
[Setup]
AppName=Film Lab
AppVersion=1.0.3
AppPublisher=Fishbone
AppCopyright=© 2026 Fishbone
DefaultDirName={autopf}\Film Lab
DefaultGroupName=Film Lab
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=FilmLab-1.0.3-Windows-x64-Setup
SetupIconFile=..\film-lab.ico
UninstallDisplayIcon={app}\Film Lab.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\build-win\dist\Film Lab.exe"; DestDir: "{app}"
Source: "..\filmlab\luts\CREDITS.txt"; DestDir: "{app}"; DestName: "Credits.txt"

[Icons]
Name: "{group}\Film Lab"; Filename: "{app}\Film Lab.exe"
Name: "{autodesktop}\Film Lab"; Filename: "{app}\Film Lab.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Film Lab.exe"; Description: "Start Film Lab"; Flags: nowait postinstall skipifsilent
