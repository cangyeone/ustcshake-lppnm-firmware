# Keep workstation paths out of released compiler diagnostics/assertion strings.
from pathlib import Path
Import('env')
env.Append(CCFLAGS=['-ffile-prefix-map='+str(Path.home())+'=/build/home'])
