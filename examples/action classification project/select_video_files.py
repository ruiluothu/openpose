import os
import glob
import shutil

# copyfile(src, dst)

path = r'C:\Users\Rui\Downloads'
os.chdir(path)

if not os.path.exists('select walking'):
    os.mkdir('select walking')

[shutil.copy(file, os.path.join(path, 'select walking')) for file in glob.glob(os.path.join(path, 'walking', '*.avi')) if
 'd1' in file or 'd2' in file]

# if not os.path.exists('select running'):
#     os.mkdir('select running')
#
# [shutil.copy(file, os.path.join(path, 'select running')) for file in glob.glob(os.path.join(path, 'running', '*.avi')) if 'd1' in file or 'd2' in file]
