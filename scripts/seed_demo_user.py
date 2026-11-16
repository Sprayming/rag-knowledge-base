"""演示账号初始化脚本（后端启动时也会自动执行，本脚本用于手工补建/换参数）。

演示账号不能走注册接口：注册会新建空租户，看不到已索引的语料。
默认挂到首个 super_admin 所在租户，角色为普通用户。

用法：
    python scripts/seed_demo_user.py
    python scripts/seed_demo_user.py --username demo2 --password demo123456 --tenant 7e990ab9
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tenant.auth import ensure_demo_user  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="demo")
    parser.add_argument("--password", default="demo123456")
    parser.add_argument("--tenant", default=None, help="默认自动取 super_admin 所在租户")
    args = parser.parse_args()

    tenant_id = ensure_demo_user(args.username, args.password, args.tenant)
    if not tenant_id:
        raise SystemExit("库里还没有 super_admin，请先在页面上注册一个账号，再运行本脚本。")
    print(f"演示账号已就绪: {args.username} / {args.password} (tenant={tenant_id}, role=user)")


if __name__ == "__main__":
    main()
