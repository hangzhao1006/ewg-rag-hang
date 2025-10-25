

#!/bin/bash
set -e

echo "Container is running!!!"
echo "Architecture: $(uname -m)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Environment ready (system python)."

# drop into interactive shell so you can run your pipeline manually
exec /bin/bash
