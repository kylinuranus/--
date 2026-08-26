$ErrorActionPreference = "Stop"

$backupPath = Join-Path $PSScriptRoot "proxy_settings_backup.json"
if (!(Test-Path $backupPath)) {
    throw "Backup file not found: $backupPath"
}

$backup = Get-Content $backupPath -Raw | ConvertFrom-Json
$reg = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings"

$proxyEnable = 0
if ($null -ne $backup.ProxyEnable) {
    $proxyEnable = [int]$backup.ProxyEnable
}
Set-ItemProperty -Path $reg -Name ProxyEnable -Value $proxyEnable

if ($null -ne $backup.ProxyServer -and $backup.ProxyServer -ne "") {
    Set-ItemProperty -Path $reg -Name ProxyServer -Value ([string]$backup.ProxyServer)
} else {
    Remove-ItemProperty -Path $reg -Name ProxyServer -ErrorAction SilentlyContinue
}

if ($null -ne $backup.ProxyOverride -and $backup.ProxyOverride -ne "") {
    Set-ItemProperty -Path $reg -Name ProxyOverride -Value ([string]$backup.ProxyOverride)
} else {
    Remove-ItemProperty -Path $reg -Name ProxyOverride -ErrorAction SilentlyContinue
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class InternetOption {
    [DllImport("wininet.dll", SetLastError=true)]
    public static extern bool InternetSetOption(IntPtr hInternet, int dwOption, IntPtr lpBuffer, int dwBufferLength);
}
"@
[InternetOption]::InternetSetOption([IntPtr]::Zero, 39, [IntPtr]::Zero, 0) | Out-Null
[InternetOption]::InternetSetOption([IntPtr]::Zero, 37, [IntPtr]::Zero, 0) | Out-Null

Write-Output "Proxy restored from backup."
