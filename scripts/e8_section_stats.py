"""Per-variant content stats for the E8 master log (run on the cloud)."""

import re
import sys

log = open("/root/autodl-tmp/core-mas/run_e8/master.log",
           encoding="utf-8", errors="replace").read()
marks = [(m.start(), m.group(1), m.group(2))
         for m in re.finditer(r"=== variant ALLOC=(\S+) FAULT=(\S+)", log)]
print("total markers:", len(marks))
for i, (pos, a, f) in enumerate(marks[-4:]):
    end = marks[-4 + i + 1][0] if i < 3 else len(log)
    seg = log[pos:end]
    print(f"{a}/{f}: errors={seg.count('!!!error')} "
          f"judge_T={seg.count('Judge reward: True')} "
          f"judge_F={seg.count('Judge reward: False')} "
          f"completed={seg.count('COMPLETED')} "
          f"len={len(seg)}")
if "--dump-last" in sys.argv:
    print("=== LAST SECTION DUMP ===")
    print(repr(log[marks[-1][0]:marks[-1][0] + 1200]))
