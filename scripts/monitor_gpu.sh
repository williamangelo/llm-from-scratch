#!/usr/bin/env bash
# Watch GPU memory usage (used and total) with nvidia-smi
watch -n 1 'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | awk -F", " "{printf \"GPU %s (%s): %s / %s MB (%.1f%%)\n\", \$1, \$2, \$3, \$4, (\$3/\$4)*100}"'
