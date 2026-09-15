"""收藏夹接口测试：覆盖 CRUD、多对多关系、幂等与软删除。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


def saved_note(db_path):
    NoteStore(db_path).save({
        "note_id": "folder-note",
        "title": "Agent 学习资料",
        "content": "学习 RAG、工具调用与 Agent 工作流。",
        "source_url": "https://www.xiaohongshu.com/explore/folder-note",
        "images": [],
    })


def test_folder_crud_and_soft_delete(tmp_path):
    db_path = tmp_path / "folders.sqlite3"
    with TestClient(create_app(db_path)) as client:
        created = client.post("/folders", json={
            "name": "AI 工程", "description": "RAG 和 Agent", "color": "#4CAF50"
        })
        assert created.status_code == 201
        folder_id = created.json()["id"]
        assert created.json()["note_count"] == 0

        assert client.post("/folders", json={"name": "AI 工程"}).status_code == 409
        updated = client.put(f"/folders/{folder_id}", json={"description": "求职主线"})
        assert updated.json()["description"] == "求职主线"

        archived = client.delete(f"/folders/{folder_id}")
        assert archived.json()["status"] == "archived"
        assert client.get("/folders").json()["items"] == []
        assert len(client.get("/folders?include_archived=true").json()["items"]) == 1


def test_note_folder_relation_is_many_to_many_and_idempotent(tmp_path):
    db_path = tmp_path / "folders.sqlite3"
    saved_note(db_path)
    with TestClient(create_app(db_path)) as client:
        first = client.post("/folders", json={"name": "AI 工程"}).json()
        second = client.post("/folders", json={"name": "求职准备"}).json()

        added = client.post(f"/notes/folder-note/folders/{first['id']}").json()
        repeated = client.post(f"/notes/folder-note/folders/{first['id']}").json()
        client.post(f"/notes/folder-note/folders/{second['id']}")
        assert added["status"] == "added"
        assert repeated["status"] == "existing"
        assert len(client.get("/notes/folder-note/folders").json()["items"]) == 2
        assert client.get(f"/folders/{first['id']}").json()["note_count"] == 1
        assert client.get(f"/folders/{first['id']}/notes").json()["items"][0]["note_id"] == "folder-note"

        removed = client.delete(f"/notes/folder-note/folders/{first['id']}")
        assert removed.json()["status"] == "removed"
        assert client.delete(f"/notes/folder-note/folders/{first['id']}").status_code == 404


def test_cannot_add_note_to_archived_folder(tmp_path):
    db_path = tmp_path / "folders.sqlite3"
    saved_note(db_path)
    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "旧收藏夹"}).json()
        client.delete(f"/folders/{folder['id']}")
        response = client.post(f"/notes/folder-note/folders/{folder['id']}")
        assert response.status_code == 404
