"""本地试用启动器：用 web 分支代码 + 主仓真实数据库。

直接 `python serve_real.py` 即可，不需要 uvicorn 命令行参数。
前端访问 http://127.0.0.1:8011/ui
"""
import os
from pathlib import Path

# 锁定工作目录到本 worktree，保证 load_dotenv 读到本目录的 .env
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()  # 载入 DEEPSEEK_API_KEY 等

from app.main import create_app

# 主仓真实笔记库（含评论、图片资产）
REAL_DB = r"C:/xhs/xhs-backend/data/notes.sqlite3"

app = create_app(db_path=REAL_DB)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8011, log_level="info")
