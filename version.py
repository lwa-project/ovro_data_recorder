import os
import logging
from subprocess import check_call, check_output, CalledProcessError

__all__ = ['version',]

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

# Setup logging
log = logging.getLogger(__name__)

# Query the git repo info.  If this fails make a note of it and yield an
# unknown version
version = 'unknown'
try:
    git_branch = check_output(['git', 'branch', '--show-current'],
                              cwd=REPO_PATH)
    git_branch = git_branch.decode().strip().rstrip()
    git_hash = check_output(['git', 'log', '-n', '1', '--pretty=format:%H'],
                            cwd=REPO_PATH)
    git_hash = git_hash.decode().strip().rstrip()
    
    git_dirty = 0
    try:
        check_call(['git', 'diff-index', '--quiet', '--cached', 'HEAD', '--'],
                   cwd=REPO_PATH)
    except CalledProcessError:
        git_dirty += 1
    try:
        check_call(['git', 'diff-files', '--quiet'],
                   cwd=REPO_PATH)
    except CalledProcessError:
        git_dirty += 1
        
    version = git_branch+'@'+git_hash[:7]
    if git_dirty > 0:
        version += ' (dirty)'
        
except CalledProcessError as e:
    log.warning("Failed to determine git repo versioning - %s", str(e))


if __name__ == '__main__':
    print(version)
