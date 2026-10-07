# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows native collision export save dialog."""
import ctypes
from ctypes import wintypes
import re
from pathlib import Path

def collision_filename(model_name):
    name = Path(str(model_name).replace('\\', '/')).stem
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .') or 'splat-model'
    return name[:180] + '-collision.obj'

def format_path(filename, format_index):
    suffix = ['.obj','.voxel.json','.glb'][format_index]
    name = str(filename)
    if name.lower().endswith(suffix):
        return name
    for extension in ['.voxel.json','.obj','.glb','.json']:
        if name.lower().endswith(extension):
            name = name[:-len(extension)]
            break
    return name+suffix

def choose_export_path(default_name, voxel=False, glb=False):
    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ('lStructSize', wintypes.DWORD), ('hwndOwner', wintypes.HWND),
            ('hInstance', wintypes.HINSTANCE), ('lpstrFilter', wintypes.LPCWSTR),
            ('lpstrCustomFilter', wintypes.LPWSTR), ('nMaxCustFilter', wintypes.DWORD),
            ('nFilterIndex', wintypes.DWORD), ('lpstrFile', wintypes.LPWSTR),
            ('nMaxFile', wintypes.DWORD), ('lpstrFileTitle', wintypes.LPWSTR),
            ('nMaxFileTitle', wintypes.DWORD), ('lpstrInitialDir', wintypes.LPCWSTR),
            ('lpstrTitle', wintypes.LPCWSTR), ('Flags', wintypes.DWORD),
            ('nFileOffset', wintypes.WORD), ('nFileExtension', wintypes.WORD),
            ('lpstrDefExt', wintypes.LPCWSTR), ('lCustData', wintypes.LPARAM),
            ('lpfnHook', ctypes.c_void_p), ('lpTemplateName', wintypes.LPCWSTR),
            ('pvReserved', ctypes.c_void_p), ('dwReserved', wintypes.DWORD),
            ('FlagsEx', wintypes.DWORD)]
    filename = ctypes.create_unicode_buffer(default_name, 32768)
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.GetActiveWindow.restype = wintypes.HWND
    dialog = OPENFILENAMEW()
    dialog.lStructSize = ctypes.sizeof(dialog)
    dialog.hwndOwner = user32.GetActiveWindow()
    dialog.lpstrFilter = ('Wavefront OBJ mesh (*.obj)\0*.obj\0'
                         'PlayCanvas voxel header (*.voxel.json)\0*.voxel.json\0'
                         'glTF binary mesh (*.glb)\0*.glb\0\0')
    dialog.nFilterIndex = 2 if voxel else 3 if glb else 1
    dialog.lpstrFile = ctypes.cast(filename, wintypes.LPWSTR)
    dialog.nMaxFile = len(filename)
    dialog.lpstrTitle = 'Export collision mesh'
    # Apply the chosen filter's suffix ourselves, including the compound voxel suffix.
    dialog.lpstrDefExt = None
    # Explorer UI, overwrite prompt, existing destination directory, keep cwd.
    dialog.Flags = 0x00080000 | 0x00000002 | 0x00000800 | 0x00000008
    comdlg32 = ctypes.WinDLL('comdlg32', use_last_error=True)
    comdlg32.GetSaveFileNameW.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
    comdlg32.GetSaveFileNameW.restype = wintypes.BOOL
    if comdlg32.GetSaveFileNameW(ctypes.byref(dialog)):
        format_index = dialog.nFilterIndex-1
        selected = format_path(filename.value,format_index)
        if selected != filename.value and Path(selected).exists():
            user32.MessageBoxW.argtypes = [wintypes.HWND,wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.UINT]
            user32.MessageBoxW.restype = ctypes.c_int
            if user32.MessageBoxW(dialog.hwndOwner,f'{selected}\n\nReplace this file?',
                                 'Confirm export overwrite',0x00000004|0x00000030|0x00000100) != 6:
                return None
        return selected, format_index
    error = comdlg32.CommDlgExtendedError()
    if error:
        raise RuntimeError(f'Save dialog failed (Windows error {error:#x}).')
    return None
