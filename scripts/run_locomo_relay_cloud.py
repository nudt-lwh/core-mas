# -*- coding: utf-8 -*-
"""LoCoMo 协议内重跑（含 source_relay）——云端一键启动。

推送更新后的 locomo_emem_fault.py + locomo_fault.py 到 AutoDL，
然后在云端以 deepseek-chat 跑 5 会话 x 10 题 x 4 策略 x (s0, agent_c)，
输出到 /root/autodl-tmp/core-mas/run_e7_relay。

用法（本地）：python scripts/run_locomo_relay_cloud.py
"""
import json
import os
import sys
import tempfile
import time

import paramiko

ROOT = r"D:\SCI\core-mas-code"
CRED = json.load(open(os.path.join(tempfile.gettempdir(), "autodl_ssh.json"), encoding="utf-8"))
KEY = open(os.path.join(os.path.dirname(ROOT), "DeepSeek_key.txt"), encoding="utf-8").read().strip()

FILES = {
    "src/core_mas/locomo_emem_fault.py": "/root/autodl-tmp/core-mas/src/core_mas/locomo_emem_fault.py",
    "src/core_mas/locomo_fault.py": "/root/autodl-tmp/core-mas/src/core_mas/locomo_fault.py",
}

REMOTE_CMD = (
    "cd /root/autodl-tmp/core-mas && "
    "nohup env PYTHONPATH=src:emem LLM_BASE_URL=https://api.deepseek.com/v1 "
    "LLM_MODEL=deepseek-chat LLM_API_KEY='" + KEY + "' "
    "/root/miniconda3/bin/python -m core_mas.locomo_emem_fault "
    "--locomo locomo10.json --emem-repo emem "
    "--config emem/evaluation/locomo/config.yaml "
    "--out-dir run_e7_relay --convs 5 --questions-per-conv 10 "
    "--budget-ratio 0.6 --victims s0 agent_c --include-relay "
    "> run_e7_relay.log 2>&1 </dev/null & echo E7-RELAY-LAUNCHED"
)


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(CRED["host"], port=CRED["port"], username=CRED["user"],
                   password=CRED["password"], timeout=30)
    for local_rel, remote in FILES.items():
        import base64
        data = open(os.path.join(ROOT, local_rel), "rb").read()
        b64 = base64.b64encode(data).decode("ascii")
        cmd = f"echo {b64} | base64 -d > {remote} && wc -c {remote}"
        stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
        out = stdout.read().decode()
        err = stderr.read().decode()
        print(local_rel, "->", out.strip(), err.strip()[:120])
    stdin, stdout, stderr = client.exec_command(REMOTE_CMD, timeout=30)
    print(stdout.read().decode().strip())
    err = stderr.read().decode().strip()
    if err:
        print("STDERR:", err[:300])
    client.close()
    print("launched; log at cloud: run_e7_relay.log")


if __name__ == "__main__":
    main()
