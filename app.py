from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import sqlite3, os, uuid, time, json, hashlib, secrets, shutil, asyncio, threading, io, subprocess, sys
from typing import Optional
from datetime import datetime
from pathlib import Path

# ── ensure SigLIP2 dependencies are installed ────────────────────────────────
def _ensure_deps():
    deps = []
    try:
        import sentencepiece
    except ImportError:
        deps.append("sentencepiece")
    try:
        import google.protobuf
    except ImportError:
        deps.append("protobuf")
    try:
        import clip
    except ImportError:
        deps.append("openai-clip")
    try:
        import cv2
    except ImportError:
        deps.append("opencv-python-headless")
    if deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps, check=True)
        # Force reimport of newly installed packages in this process
        # so background threads (backfill, SigLIP loader) don't hit ImportError
        import importlib
        for mod in ["sentencepiece", "google.protobuf"]:
            try:
                importlib.import_module(mod)
            except Exception:
                pass

_ensure_deps()

# ── CONFIG ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "findit-dev-secret")
DB_PATH    = "/app/data/findit.db"
IMG_DIR    = "/app/data/images"
TOKEN_TTL  = 60 * 60 * 24 * 30  # 30 days

Path(IMG_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FindIt API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

# ── PUB/SUB BROKER ────────────────────────────────────────────────────────────
class Broker:
    def __init__(self):
        self.listeners: dict[str, list[asyncio.Queue]] = {}  # channel → queues

    def subscribe(self, channel: str) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=50)
        self.listeners.setdefault(channel, []).append(q)
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue):
        lst = self.listeners.get(channel, [])
        if q in lst: lst.remove(q)

    def publish(self, channel: str, data: dict):
        msg = f"data: {json.dumps(data)}\n\n"
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        def _put(queues):
            for q in list(queues):
                try: q.put_nowait(msg)
                except asyncio.QueueFull: pass

        def _send():
            _put(self.listeners.get(channel, []))
            # Forward post/comment events to "all" feed listeners,
            # but NOT user-specific or admin events (those are private)
            if channel == "all":
                pass  # already sent above
            elif not channel.startswith("user:") and not channel.startswith("dm:") and channel != "admin":
                _put(self.listeners.get("all", []))

        if loop and loop.is_running():
            # Called from a background thread — schedule on the event loop
            loop.call_soon_threadsafe(_send)
        else:
            _send()

broker = Broker()


# ── DATABASE ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)  # wait up to 30s for locks
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")     # 30s busy timeout in ms
    return conn

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS profiles (
            id         TEXT PRIMARY KEY,
            uid        TEXT UNIQUE NOT NULL,
            name       TEXT NOT NULL,
            initials   TEXT NOT NULL,
            color      TEXT NOT NULL DEFAULT '#5b8dff',
            role       TEXT NOT NULL DEFAULT 'user',
            is_banned  INTEGER NOT NULL DEFAULT 0,
            points     INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS passwords (
            user_id    TEXT PRIMARY KEY REFERENCES profiles(id),
            hash       TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL REFERENCES profiles(id),
            expires_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS posts (
            id          TEXT PRIMARY KEY,
            author_id   TEXT NOT NULL REFERENCES profiles(id),
            title       TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            location    TEXT NOT NULL,
            category    TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'found',
            image_url   TEXT,
            ai_caption  TEXT,
            is_deleted  INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS comments (
            id         TEXT PRIMARY KEY,
            post_id    TEXT NOT NULL REFERENCES posts(id),
            author_id  TEXT NOT NULL REFERENCES profiles(id),
            parent_id  TEXT REFERENCES comments(id),
            body       TEXT NOT NULL DEFAULT '',
            image_url  TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS mod_log (
            id         TEXT PRIMARY KEY,
            admin_id   TEXT NOT NULL REFERENCES profiles(id),
            action     TEXT NOT NULL,
            target_id  TEXT,
            post_id    TEXT,
            note       TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS admin_requests (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL REFERENCES profiles(id),
            email      TEXT NOT NULL,
            name       TEXT NOT NULL,
            role_title TEXT NOT NULL,
            reason     TEXT NOT NULL,
            id_image_url TEXT,
            status     TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL REFERENCES profiles(id),
            admin_id   TEXT NOT NULL REFERENCES profiles(id),
            note       TEXT NOT NULL DEFAULT '',
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS reports (
            id         TEXT PRIMARY KEY,
            post_id    TEXT NOT NULL REFERENCES posts(id),
            reporter_id TEXT NOT NULL REFERENCES profiles(id),
            reason     TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(post_id, reporter_id, reason)
        );
        CREATE TABLE IF NOT EXISTS dms (
            id          TEXT PRIMARY KEY,
            sender_id   TEXT NOT NULL REFERENCES profiles(id),
            receiver_id TEXT NOT NULL REFERENCES profiles(id),
            body        TEXT NOT NULL DEFAULT '',
            image_url   TEXT,
            read        INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS comment_votes (
            id          TEXT PRIMARY KEY,
            comment_id  TEXT NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
            user_id     TEXT NOT NULL REFERENCES profiles(id),
            vote        INTEGER NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(comment_id, user_id)
        );
    """)
    db.commit()
    db.close()

init_db()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return hashlib.sha256((pw + SECRET_KEY).encode()).hexdigest()

def make_token() -> str:
    return secrets.token_hex(32)

COLORS = ["#5b8dff","#22c97a","#a084f5","#ff6b6b","#f5a623","#4da6ff","#e8385a","#00c9a7"]
def pick_color(uid: str) -> str:
    return COLORS[sum(ord(c) for c in uid) % len(COLORS)]

def get_current_user(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "): return None
    token = auth[7:]
    db  = get_db()
    row = db.execute(
        "SELECT p.* FROM sessions s JOIN profiles p ON p.id=s.user_id "
        "WHERE s.token=? AND s.expires_at>?", (token, int(time.time()))
    ).fetchone()
    db.close()
    return dict(row) if row else None

def require_user(request: Request):
    u = get_current_user(request)
    if not u: raise HTTPException(401, "Not authenticated")
    return u

def require_admin(request: Request):
    u = require_user(request)
    if u["role"] not in ("admin","super_admin"): raise HTTPException(403, "Not admin")
    return u

def create_session(user_id: str) -> str:
    token   = make_token()
    expires = int(time.time()) + TOKEN_TTL
    db = get_db()
    db.execute("INSERT INTO sessions (token,user_id,expires_at) VALUES (?,?,?)",
               (token, user_id, expires))
    db.commit()
    db.close()
    return token

# ── AUTH ──────────────────────────────────────────────────────────────────────
@app.post("/auth/register")
async def register(data: dict):
    uid = data.get("uid","").strip().lower()
    pw  = data.get("password","").strip()
    if not uid or not pw:
        raise HTTPException(400, "Username and password required")
    if len(uid) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(pw) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if not uid.replace("_","").isalnum():
        raise HTTPException(400, "Username can only contain letters, numbers, underscores")

    db = get_db()
    if db.execute("SELECT 1 FROM profiles WHERE uid=?", (uid,)).fetchone():
        db.close()
        raise HTTPException(409, "Username already taken")

    pid      = str(uuid.uuid4())
    name     = data.get("name", uid).strip() or uid
    initials = name[:2].upper()
    color    = pick_color(uid)

    # First user ever registered becomes super_admin automatically
    user_count = db.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    role = 'super_admin' if user_count == 0 else 'user'

    db.execute("INSERT INTO profiles (id,uid,name,initials,color,role) VALUES (?,?,?,?,?,?)",
               (pid, uid, name, initials, color, role))
    db.execute("INSERT INTO passwords (user_id,hash) VALUES (?,?)",
               (pid, hash_password(pw)))
    db.commit()

    profile = dict(db.execute("SELECT * FROM profiles WHERE id=?", (pid,)).fetchone())
    db.close()
    token = create_session(pid)
    return {"token": token, "profile": profile}

@app.post("/auth/register-with-id")
async def register_with_id(
    uid:      str        = Form(...),
    password: str        = Form(...),
    name:     str        = Form(""),
    id_file:  UploadFile = File(None),
):
    """
    Register a new user and optionally verify their university ID.
    Account is created immediately. If an ID image is provided, the badge
    check runs in a background thread and is pushed to the client via SSE
    so the response is never blocked by slow model loading.
    Returns: {token, profile, badge: 'pending'|'none'}
    """
    uid = uid.strip().lower()
    pw  = password.strip()
    if not uid or not pw:
        raise HTTPException(400, "Username and password required")
    if len(uid) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(pw) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if not uid.replace("_","").isalnum():
        raise HTTPException(400, "Username can only contain letters, numbers, underscores")

    db = get_db()
    if db.execute("SELECT 1 FROM profiles WHERE uid=?", (uid,)).fetchone():
        db.close()
        raise HTTPException(409, "Username already taken")

    pid        = str(uuid.uuid4())
    name       = (name.strip() or uid)
    initials   = name[:2].upper()
    color      = pick_color(uid)
    user_count = db.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    role_val   = "super_admin" if user_count == 0 else "user"

    # Read ID image bytes before DB insert (file stream can't be read in a thread)
    img_bytes = None
    has_id_file = id_file and id_file.filename
    if has_id_file:
        try:
            img_bytes = await id_file.read()
            if not img_bytes:
                img_bytes = None
        except Exception:
            img_bytes = None

    # Create account with badge='none' immediately — don't block on ID check
    db.execute(
        "INSERT INTO profiles (id,uid,name,initials,color,role,badge) VALUES (?,?,?,?,?,?,?)",
        (pid, uid, name, initials, color, role_val, "none")
    )
    db.execute("INSERT INTO passwords (user_id,hash) VALUES (?,?)",
               (pid, hash_password(pw)))
    db.commit()
    profile = dict(db.execute("SELECT * FROM profiles WHERE id=?", (pid,)).fetchone())
    db.close()
    token = create_session(pid)

    # Run ID verification in background — push badge via SSE when done
    if img_bytes:
        def _verify_and_push(user_uid: str, user_id: str, image_bytes: bytes):
            try:
                result = _siglip_check_id(image_bytes)
                print(f"[register-bg] uid={user_uid} is_id={result.get('is_id')} role={result.get('detected_role')}")
                if result.get("is_id"):
                    detected = result.get("detected_role", "unknown")
                    badge    = detected if detected in ("student", "staff") else "verified"
                else:
                    badge = "none"

                if badge != "none":
                    db2 = get_db()
                    db2.execute("UPDATE profiles SET badge=? WHERE id=?", (badge, user_id))
                    db2.commit()
                    db2.close()
                    print(f"[register-bg] ✅ badge={badge} → uid={user_uid}")
                    # Push to the user's SSE channel so the frontend updates live
                    broker.publish(f"user:{user_uid}", {
                        "type":    "badge_granted",
                        "badge":   badge,
                        "message": f"{'Student' if badge == 'student' else 'Staff' if badge == 'staff' else 'Verified'} badge added to your profile!",
                    })
            except Exception as e:
                print(f"[register-bg] id check error: {e}")

        threading.Thread(
            target=_verify_and_push,
            args=(uid, pid, img_bytes),
            daemon=True
        ).start()
        badge_status = "pending"
    else:
        badge_status = "none"

    return {"token": token, "profile": profile, "badge": badge_status}

@app.post("/auth/login")
async def login(data: dict):
    uid = data.get("uid","").strip().lower()
    pw  = data.get("password","").strip()
    if not uid or not pw:
        raise HTTPException(400, "Username and password required")

    db      = get_db()
    profile = db.execute("SELECT * FROM profiles WHERE uid=?", (uid,)).fetchone()
    if not profile:
        db.close()
        raise HTTPException(401, "Wrong username or password")

    pw_row = db.execute("SELECT hash FROM passwords WHERE user_id=?",
                        (profile["id"],)).fetchone()
    db.close()
    if not pw_row or pw_row["hash"] != hash_password(pw):
        raise HTTPException(401, "Wrong username or password")
    if profile["is_banned"]:
        raise HTTPException(403, "Account banned")

    token = create_session(profile["id"])
    return {"token": token, "profile": dict(profile)}

@app.post("/auth/logout")
async def logout(request: Request):
    auth = request.headers.get("Authorization","")
    if auth.startswith("Bearer "):
        db = get_db()
        db.execute("DELETE FROM sessions WHERE token=?", (auth[7:],))
        db.commit()
        db.close()
    return {"ok": True}

@app.get("/auth/me")
async def get_me(user=Depends(require_user)):
    if user.get("is_banned"):
        raise HTTPException(403, "Account banned")
    return {"profile": user}

# ── PROFILES ──────────────────────────────────────────────────────────────────
@app.get("/profiles/{uid}/stats")
async def profile_stats(uid: str):
    db = get_db()
    p  = db.execute("SELECT * FROM profiles WHERE uid=?", (uid,)).fetchone()
    if not p: raise HTTPException(404, "User not found")
    posts    = db.execute("SELECT COUNT(*) FROM posts WHERE author_id=? AND is_deleted=0", (p["id"],)).fetchone()[0]
    comments = db.execute("SELECT COUNT(*) FROM comments WHERE author_id=?", (p["id"],)).fetchone()[0]
    db.close()
    return {"postCount": posts, "commentCount": comments, "points": posts*50 + comments*10, "role": p["role"], "badge": p["badge"] if "badge" in p.keys() else "none"}

@app.get("/profiles/{uid}/posts")
async def profile_posts(uid: str):
    db   = get_db()
    rows = db.execute("""
        SELECT p.*, pr.uid as author_uid, pr.name as author_name,
               pr.initials as author_initials, pr.color as author_color,
               (SELECT COUNT(*) FROM comments c WHERE c.post_id=p.id) as comment_count
        FROM posts p JOIN profiles pr ON pr.id=p.author_id
        WHERE pr.uid=? AND p.is_deleted=0 ORDER BY p.created_at DESC
    """, (uid,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ── POSTS ─────────────────────────────────────────────────────────────────────
@app.get("/posts")
async def get_posts():
    db   = get_db()
    rows = db.execute("""
        SELECT p.*, pr.uid as author_uid, pr.name as author_name,
               pr.initials as author_initials, pr.color as author_color,
               pr.role as author_role, pr.is_banned as author_banned,
               pr.badge as author_badge,
               (SELECT COUNT(*) FROM comments c WHERE c.post_id=p.id) as comment_count
        FROM posts p JOIN profiles pr ON pr.id=p.author_id
        WHERE p.is_deleted=0 ORDER BY p.created_at DESC
    """).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.get("/posts/since")
async def get_posts_since(ts: str = ""):
    db   = get_db()
    sql  = ("SELECT p.*, pr.uid as author_uid, pr.name as author_name,"
            "pr.initials as author_initials, pr.color as author_color,"
            "pr.role as author_role, pr.is_banned as author_banned,"
            "pr.badge as author_badge,"
            "(SELECT COUNT(*) FROM comments c WHERE c.post_id=p.id) as comment_count "
            "FROM posts p JOIN profiles pr ON pr.id=p.author_id "
            "WHERE p.is_deleted=0 AND p.created_at > ? ORDER BY p.created_at DESC")
    rows = db.execute(sql, (ts,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.get("/stream")
async def stream(request: Request, channel: str = "all"):
    """SSE endpoint. channel='all' for posts, 'dm:{uid}' for DM notifications."""
    q = broker.subscribe(channel)
    async def event_generator():
        try:
            # Send a heartbeat immediately to confirm connection
            yield "data: {\"type\":\"connected\"}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # SSE comment = keepalive
        finally:
            broker.unsubscribe(channel, q)
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



@app.post("/posts")
async def create_post(request: Request, user=Depends(require_user)):
    data = await request.json()
    pid  = str(uuid.uuid4())
    db   = get_db()
    db.execute(
        "INSERT INTO posts (id,author_id,title,description,location,category,status,image_url) VALUES (?,?,?,?,?,?,?,?)",
        (pid, user["id"], data["title"], data.get("description",""),
         data["location"], data["category"], data.get("status","found"), data.get("image_url"))
    )
    db.commit()
    row = db.execute("""
        SELECT p.*, pr.uid as author_uid, pr.name as author_name,
               pr.initials as author_initials, pr.color as author_color, 0 as comment_count
        FROM posts p JOIN profiles pr ON pr.id=p.author_id WHERE p.id=?
    """, (pid,)).fetchone()
    db.close()
    post_data = dict(row)
    broker.publish("all", {"type": "new_post", "post": post_data})
    return post_data

@app.patch("/posts/{post_id}")
async def update_post(post_id: str, request: Request, user=Depends(require_user)):
    data = await request.json()
    db   = get_db()
    post = db.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    if not post: raise HTTPException(404)
    if post["author_id"] != user["id"] and user["role"] not in ("admin","super_admin"):
        raise HTTPException(403)
    fields = {k:v for k,v in data.items() if k in ("title","description","status","image_url","is_deleted")}
    fields["updated_at"] = datetime.utcnow().isoformat()
    sets = ", ".join(f"{k}=?" for k in fields)
    db.execute(f"UPDATE posts SET {sets} WHERE id=?", (*fields.values(), post_id))
    db.commit()
    db.close()
    return {"ok": True}

@app.delete("/posts/{post_id}")
async def delete_post(post_id: str, user=Depends(require_user)):
    db     = get_db()
    post   = db.execute("SELECT p.*, pr.role as author_role FROM posts p JOIN profiles pr ON pr.id=p.author_id WHERE p.id=?", (post_id,)).fetchone()
    if not post: raise HTTPException(404)
    is_own   = post["author_id"] == user["id"]
    is_admin = user["role"] in ("admin","super_admin")
    is_super = user["role"] == "super_admin"
    author_is_admin = post["author_role"] in ("admin","super_admin")
    author_is_super = post["author_role"] == "super_admin"
    # owner can always delete own post
    if not is_own and not is_admin: raise HTTPException(403)
    # admin can't delete other admin or super posts — only super can
    if not is_own and author_is_admin and not is_super:
        raise HTTPException(403, "Only super admin can delete admin/super posts")
    db.execute("UPDATE posts SET is_deleted=1 WHERE id=?", (post_id,))
    db.commit()
    db.close()
    return {"ok": True}

# ── COMMENTS ──────────────────────────────────────────────────────────────────
@app.get("/posts/{post_id}/comments")
async def get_comments(post_id: str, request: Request):
    db      = get_db()
    # Try to identify current user for my_vote
    user_id = None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        tok = auth[7:]
        row = db.execute("SELECT user_id FROM sessions WHERE token=?", (tok,)).fetchone()
        if row: user_id = row["user_id"]

    rows = db.execute("""
        SELECT c.id, c.post_id, c.author_id, c.parent_id, c.body, c.image_url, c.created_at,
               pr.uid as author_uid, pr.name as author_name,
               pr.initials as author_initials, pr.color as author_color,
               COALESCE(SUM(CASE WHEN v.vote=1  THEN 1 ELSE 0 END),0) as upvotes,
               COALESCE(SUM(CASE WHEN v.vote=-1 THEN 1 ELSE 0 END),0) as downvotes,
               COALESCE(SUM(v.vote),0) as net_votes
        FROM comments c
        JOIN profiles pr ON pr.id=c.author_id
        LEFT JOIN comment_votes v ON v.comment_id=c.id
        WHERE c.post_id=?
        GROUP BY c.id
        ORDER BY c.created_at ASC
    """, (post_id,)).fetchall()

    my_votes = {}
    if user_id:
        mv = db.execute(
            "SELECT comment_id, vote FROM comment_votes WHERE user_id=?", (user_id,)
        ).fetchall()
        my_votes = {r["comment_id"]: r["vote"] for r in mv}
    db.close()

    result = []
    for r in rows:
        d = dict(r)
        d["author"] = {"uid": d.pop("author_uid"), "name": d.pop("author_name"),
                       "initials": d.pop("author_initials"), "color": d.pop("author_color")}
        d["my_vote"] = my_votes.get(d["id"], 0)
        result.append(d)
    return result

@app.post("/posts/{post_id}/comments")
async def create_comment(post_id: str, request: Request, user=Depends(require_user)):
    data = await request.json()
    cid  = str(uuid.uuid4())
    db   = get_db()
    db.execute(
        "INSERT INTO comments (id,post_id,author_id,parent_id,body,image_url) VALUES (?,?,?,?,?,?)",
        (cid, post_id, user["id"], data.get("parent_id"), data.get("body",""), data.get("image_url"))
    )
    db.commit()
    row = db.execute("""
        SELECT c.*, pr.uid as author_uid, pr.name as author_name,
               pr.initials as author_initials, pr.color as author_color
        FROM comments c JOIN profiles pr ON pr.id=c.author_id WHERE c.id=?
    """, (cid,)).fetchone()
    db.close()
    d = dict(row)
    d["author"] = {"uid": d.pop("author_uid"), "name": d.pop("author_name"),
                   "initials": d.pop("author_initials"), "color": d.pop("author_color")}
    d["net_votes"] = 0; d["my_vote"] = 0; d["upvotes"] = 0; d["downvotes"] = 0
    # Broadcast to everyone viewing this post
    broker.publish(f"post:{post_id}", {"type": "new_comment", "comment": d})
    return d

# ── IMAGE UPLOAD ──────────────────────────────────────────────────────────────
@app.patch("/comments/{comment_id}")
async def edit_comment(comment_id: str, request: Request, user=Depends(require_user)):
    comment = get_db().execute("SELECT * FROM comments WHERE id=?", (comment_id,)).fetchone()
    if not comment: raise HTTPException(404)
    if comment["author_id"] != user["id"]: raise HTTPException(403, "Only the author can edit")
    data = await request.json()
    body = data.get("body","").strip()
    if not body: raise HTTPException(400, "Body required")
    db = get_db()
    db.execute("UPDATE comments SET body=? WHERE id=?", (body, comment_id))
    db.commit(); db.close()
    return {"ok": True, "body": body}

@app.post("/comments/{comment_id}/report")
async def report_comment(comment_id: str, request: Request, user=Depends(require_user)):
    db      = get_db()
    comment = db.execute("SELECT * FROM comments WHERE id=?", (comment_id,)).fetchone()
    if not comment: raise HTTPException(404)
    if comment["author_id"] == user["id"]: raise HTTPException(403, "Cannot report own comment")
    if user["role"] in ("admin","super_admin"): raise HTTPException(403, "Admins cannot report")
    data   = await request.json()
    db.execute(
        "INSERT OR REPLACE INTO reports (id,post_id,reporter_id,reason) VALUES (?,?,?,?)",
        (str(uuid.uuid4()), comment["post_id"], user["id"], f'comment:{comment_id}:{data.get("reason","")}')
    )
    db.commit(); db.close()
    return {"ok": True}

@app.delete("/comments/{comment_id}")
async def delete_comment(comment_id: str, user=Depends(require_user)):
    db      = get_db()
    comment = db.execute("SELECT c.*, pr.role as author_role FROM comments c JOIN profiles pr ON pr.id=c.author_id WHERE c.id=?", (comment_id,)).fetchone()
    if not comment: raise HTTPException(404)
    is_own   = comment["author_id"] == user["id"]
    is_admin = user["role"] in ("admin","super_admin")
    if not is_own and not is_admin:
        raise HTTPException(403, "Cannot delete this comment")
    db.execute("DELETE FROM comments WHERE id=?", (comment_id,))
    db.commit()
    db.close()
    return {"ok": True}

@app.post("/comments/{comment_id}/vote")
async def vote_comment(comment_id: str, request: Request, user=Depends(require_user)):
    data = await request.json()
    vote = int(data.get("vote", 0))
    db   = get_db()
    if vote == 0:
        db.execute("DELETE FROM comment_votes WHERE comment_id=? AND user_id=?", (comment_id, user["id"]))
    else:
        sql = ("INSERT INTO comment_votes (id,comment_id,user_id,vote) VALUES (?,?,?,?) "
               "ON CONFLICT(comment_id,user_id) DO UPDATE SET vote=excluded.vote")
        db.execute(sql, (str(uuid.uuid4()), comment_id, user["id"], vote))
    row = db.execute(
        "SELECT COALESCE(SUM(CASE WHEN vote=1 THEN 1 ELSE 0 END),0) as upvotes,"
        "COALESCE(SUM(CASE WHEN vote=-1 THEN 1 ELSE 0 END),0) as downvotes,"
        "COALESCE(SUM(vote),0) as net_votes FROM comment_votes WHERE comment_id=?",
        (comment_id,)
    ).fetchone()
    db.commit(); db.close()
    return {"ok":True,"upvotes":row["upvotes"],"downvotes":row["downvotes"],"net_votes":row["net_votes"],"my_vote":vote}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), user=Depends(require_user)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Not an image")
    ext      = file.content_type.split("/")[1].replace("jpeg","jpg")
    filename = f"{uuid.uuid4()}.{ext}"
    with open(os.path.join(IMG_DIR, filename), "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"url": f"/images/{filename}"}

# ── DIRECT MESSAGES ──────────────────────────────────────────────────────────

@app.get("/dms/conversations")
async def get_conversations(user=Depends(require_user)):
    """Get all conversations for the current user, with latest message and unread count."""
    db   = get_db()
    rows = db.execute("""
        SELECT
            p.id, p.uid, p.name, p.initials, p.color,
            (SELECT body FROM dms WHERE
                (sender_id=? AND receiver_id=p.id) OR
                (sender_id=p.id AND receiver_id=?)
                ORDER BY created_at DESC LIMIT 1) as last_msg,
            (SELECT created_at FROM dms WHERE
                (sender_id=? AND receiver_id=p.id) OR
                (sender_id=p.id AND receiver_id=?)
                ORDER BY created_at DESC LIMIT 1) as last_at,
            (SELECT COUNT(*) FROM dms WHERE sender_id=p.id AND receiver_id=? AND read=0) as unread
        FROM profiles p
        WHERE p.id != ?
          AND EXISTS (
            SELECT 1 FROM dms WHERE
                (sender_id=? AND receiver_id=p.id) OR
                (sender_id=p.id AND receiver_id=?)
          )
        ORDER BY last_at DESC
    """, (user["id"],)*8).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.get("/dms/{other_uid}")
async def get_dm_thread(other_uid: str, user=Depends(require_user)):
    """Get all messages between current user and another user."""
    db    = get_db()
    other = db.execute("SELECT * FROM profiles WHERE uid=?", (other_uid,)).fetchone()
    if not other: raise HTTPException(404, "User not found")
    other = dict(other)
    msgs  = db.execute("""
        SELECT d.*, p.uid as sender_uid, p.initials as sender_initials, p.color as sender_color
        FROM dms d JOIN profiles p ON p.id=d.sender_id
        WHERE (d.sender_id=? AND d.receiver_id=?) OR (d.sender_id=? AND d.receiver_id=?)
        ORDER BY d.created_at ASC
    """, (user["id"], other["id"], other["id"], user["id"])).fetchall()
    # Mark received messages as read
    db.execute("UPDATE dms SET read=1 WHERE sender_id=? AND receiver_id=? AND read=0",
               (other["id"], user["id"]))
    db.commit()
    db.close()
    return {"other": other, "messages": [dict(m) for m in msgs]}

@app.post("/dms/{other_uid}")
async def send_dm(other_uid: str, request: Request, user=Depends(require_user)):
    """Send a message to another user."""
    db    = get_db()
    other = db.execute("SELECT * FROM profiles WHERE uid=?", (other_uid,)).fetchone()
    if not other: raise HTTPException(404, "User not found")
    data  = await request.json()
    body      = data.get("body","").strip()
    image_url = data.get("image_url")
    if not body and not image_url: raise HTTPException(400, "Empty message")
    mid = str(uuid.uuid4())
    db.execute("INSERT INTO dms (id,sender_id,receiver_id,body,image_url) VALUES (?,?,?,?,?)",
               (mid, user["id"], other["id"], body, image_url))
    db.commit()
    db.close()
    msg_data = {"id": mid, "sender_uid": user["uid"], "body": body, "image_url": image_url,
                "created_at": datetime.utcnow().isoformat(), "read": 0}
    # Notify recipient instantly via SSE
    broker.publish(f"dm:{other_uid}", {"type": "new_dm", "from_uid": user["uid"], "msg": msg_data})
    return msg_data

@app.get("/dms/unread/count")
async def unread_count(user=Depends(require_user)):
    db    = get_db()
    count = db.execute("SELECT COUNT(*) FROM dms WHERE receiver_id=? AND read=0", (user["id"],)).fetchone()[0]
    db.close()
    return {"count": count}

# ── ALERTS ───────────────────────────────────────────────────────────────────

@app.post("/alerts/{target_uid}")
async def send_alert(target_uid: str, request: Request, user=Depends(require_admin)):
    target = get_db().execute("SELECT * FROM profiles WHERE uid=?", (target_uid,)).fetchone()
    if not target: raise HTTPException(404)
    if target["role"] == "super_admin": raise HTTPException(403, "Cannot alert super admin")
    if target["role"] == "admin" and user["role"] != "super_admin": raise HTTPException(403, "Only super admin can alert admins")
    data   = await request.json()
    db     = get_db()
    # Count active alerts
    now    = datetime.utcnow().isoformat()
    count  = db.execute("SELECT COUNT(*) FROM alerts WHERE user_id=? AND expires_at>?", (target["id"], now)).fetchone()[0]
    mid    = str(uuid.uuid4())
    expiry = datetime.utcnow().replace(day=datetime.utcnow().day).isoformat()  # placeholder
    from datetime import timedelta
    expiry = (datetime.utcnow() + timedelta(days=30)).isoformat()
    note = data.get("note","")
    db.execute("INSERT INTO alerts (id,user_id,admin_id,note,expires_at) VALUES (?,?,?,?,?)",
               (mid, target["id"], user["id"], note, expiry))
    # Auto-ban after 5 active alerts
    new_count = count + 1
    if new_count >= 5:
        db.execute("UPDATE profiles SET is_banned=1 WHERE id=?", (target["id"],))
    db.commit()
    db.close()
    # Push alert to the target user SSE channel so the bell lights up immediately
    broker.publish(f"user:{target['uid']}", {
        "type": "alert_received",
        "note": note,
        "alert_count": new_count,
        "auto_banned": new_count >= 5,
    })
    if new_count >= 5:
        broker.publish(f"user:{target['uid']}", {"type": "banned"})
    return {"ok": True, "auto_banned": new_count >= 5}

@app.get("/alerts/{uid}")
async def get_alerts(uid: str, user=Depends(get_current_user)):
    db     = get_db()
    target = db.execute("SELECT * FROM profiles WHERE uid=?", (uid,)).fetchone()
    if not target: raise HTTPException(404)
    now    = datetime.utcnow().isoformat()
    rows   = db.execute("""
        SELECT a.*, p.uid as admin_uid FROM alerts a
        JOIN profiles p ON p.id=a.admin_id
        WHERE a.user_id=? AND a.expires_at>?
        ORDER BY a.created_at DESC
    """, (target["id"], now)).fetchall()
    db.close()
    return {"count": len(rows), "alerts": [dict(r) for r in rows]}

# ── REPORTS ──────────────────────────────────────────────────────────────────

@app.post("/reports/{post_id}")
async def report_post(post_id: str, request: Request, user=Depends(require_user)):
    if user["role"] in ("admin","super_admin"): raise HTTPException(403, "Admins cannot report posts")
    db   = get_db()
    post = db.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    if not post: raise HTTPException(404)
    if post["author_id"] == user["id"]: raise HTTPException(403, "Cannot report your own post")
    data = await request.json()
    db.execute(
        "INSERT OR REPLACE INTO reports (id,post_id,reporter_id,reason) VALUES (?,?,?,?)",
        (str(uuid.uuid4()), post_id, user["id"], data.get("reason",""))
    )
    db.commit()
    db.close()
    return {"ok": True}

@app.delete("/reports/{post_id}")
async def delete_report(post_id: str, user=Depends(require_admin)):
    db = get_db()
    db.execute("DELETE FROM reports WHERE post_id=?", (post_id,))
    db.commit()
    db.close()
    return {"ok": True}

@app.get("/reports")
async def get_reports(user=Depends(require_admin)):
    db   = get_db()
    rows = db.execute("""
        SELECT p.id, p.title, p.description, p.location, p.category, p.status,
               p.author_id, p.image_url, p.created_at,
               pr.uid as author_uid, pr.name as author_name,
               pr.initials as author_initials, pr.color as author_color,
               COUNT(r.id) as report_count,
               GROUP_CONCAT(r.reason, ', ') as reasons
        FROM reports r
        JOIN posts p ON p.id=r.post_id
        JOIN profiles pr ON pr.id=p.author_id
        WHERE p.is_deleted=0
          AND (r.reason IS NULL OR r.reason NOT LIKE 'comment:%')
        GROUP BY p.id ORDER BY report_count DESC
    """).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.get("/reports/comments")
async def get_comment_reports(user=Depends(require_admin)):
    """Return all comment reports grouped by comment (not post)."""
    db   = get_db()
    # Fetch all comment-type reports
    raw = db.execute(
        "SELECT id, reason, reporter_id, created_at FROM reports "
        "WHERE reason LIKE 'comment:%' ORDER BY created_at DESC"
    ).fetchall()

    seen_comment_ids = set()
    results = []
    for r in raw:
        parts = r['reason'].split(':', 2)  # comment:<id>:<reason_text>
        if len(parts) < 2:
            continue
        comment_id  = parts[1]
        reason_text = parts[2] if len(parts) > 2 else ''

        if comment_id in seen_comment_ids:
            continue  # deduplicate – show each comment once
        seen_comment_ids.add(comment_id)

        c = db.execute(
            "SELECT c.*, pr.uid as author_uid, pr.name as author_name, "
            "pr.initials as author_initials, pr.color as author_color, "
            "p.title as post_title, p.image_url as post_image_url "
            "FROM comments c "
            "JOIN profiles pr ON pr.id=c.author_id "
            "JOIN posts p ON p.id=c.post_id "
            "WHERE c.id=? AND p.is_deleted=0",
            (comment_id,)
        ).fetchone()
        if not c:
            continue  # comment or post deleted

        results.append({
            "report_id":         r['id'],
            "report_date":       r['created_at'],
            "reason":            f'comment:{comment_id}:{reason_text}',
            "comment_id":        comment_id,
            "comment_body":      c['body'],
            "comment_image_url": c['image_url'],
            "post_id":           c['post_id'],
            "post_title":        c['post_title'],
            "post_image_url":    c['post_image_url'],
            "author_uid":        c['author_uid'],
            "author_name":       c['author_name'],
            "author_initials":   c['author_initials'],
            "author_color":      c['author_color'],
        })
    db.close()
    return results

@app.delete("/reports/comments/{comment_id}")
async def delete_comment_report(comment_id: str, user=Depends(require_admin)):
    """Dismiss a comment report (delete comment or just clear the report)."""
    db = get_db()
    db.execute("DELETE FROM reports WHERE reason LIKE ?", (f'comment:{comment_id}:%',))
    db.commit()
    db.close()
    return {"ok": True}

# ── MOD LOG ──────────────────────────────────────────────────────────────────

@app.post("/admin/log")
async def create_log(request: Request, user=Depends(require_admin)):
    data = await request.json()
    db   = get_db()
    db.execute(
        "INSERT INTO mod_log (id,admin_id,action,target_id,post_id,note) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), user["id"], data.get("action",""),
         data.get("target_id"), data.get("post_id"), data.get("note",""))
    )
    db.commit()
    db.close()
    broker.publish("admin", {"type": "new_log"})
    return {"ok": True}

@app.get("/admin/log")
async def get_log(user=Depends(require_admin)):
    db   = get_db()
    sql  = """
        SELECT l.*,
               a.uid as admin_uid, a.name as admin_name,
               t.uid as target_uid, t.name as target_name,
               p.title as post_title
        FROM mod_log l
        JOIN profiles a ON a.id=l.admin_id
        LEFT JOIN profiles t ON t.id=l.target_id
        LEFT JOIN posts p ON p.id=l.post_id
        ORDER BY l.created_at DESC LIMIT 200
    """
    rows = db.execute(sql).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ── ADMIN ─────────────────────────────────────────────────────────────────────
@app.get("/admin/stats")
async def admin_stats(user=Depends(require_admin)):
    db    = get_db()
    posts  = db.execute("SELECT COUNT(*) FROM posts WHERE is_deleted=0").fetchone()[0]
    active = db.execute("SELECT COUNT(*) FROM posts WHERE is_deleted=0 AND status!='recovered'").fetchone()[0]
    users  = db.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    admins = db.execute("SELECT COUNT(*) FROM profiles WHERE role IN ('admin','super_admin')").fetchone()[0]
    banned = db.execute("SELECT COUNT(*) FROM profiles WHERE is_banned=1").fetchone()[0]
    pend   = db.execute("SELECT COUNT(*) FROM admin_requests WHERE status='pending'").fetchone()[0]
    reports        = db.execute("SELECT COUNT(DISTINCT post_id) FROM reports WHERE reason NOT LIKE 'comment:%'").fetchone()[0]
    comment_reports = db.execute("SELECT COUNT(DISTINCT r.reason) FROM reports r WHERE r.reason LIKE 'comment:%'").fetchone()[0]
    db.close()
    return {"totalPosts":posts,"activePosts":active,"totalUsers":users,"admins":admins,"bannedUsers":banned,"pendingRequests":pend,"reportedPosts":reports,"commentReports":comment_reports}

@app.get("/admin/requests")
async def get_admin_requests(user=Depends(require_admin)):
    db   = get_db()
    rows = db.execute("""
        SELECT r.*, p.uid as requester_uid
        FROM admin_requests r
        LEFT JOIN profiles p ON p.id=r.user_id
        WHERE r.status='pending'
        ORDER BY r.created_at DESC
    """).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.patch("/admin/requests/{req_id}")
async def review_request(req_id: str, request: Request, user=Depends(require_admin)):
    data   = await request.json()
    status = data.get("status","rejected")
    db     = get_db()
    db.execute("UPDATE admin_requests SET status=? WHERE id=?", (status, req_id))
    db.commit()
    db.close()
    broker.publish("admin", {"type": "request_reviewed", "req_id": req_id, "status": status})
    return {"ok": True}

@app.get("/admin/users")
async def get_all_users(user=Depends(require_admin)):
    db   = get_db()
    rows = db.execute("SELECT * FROM profiles ORDER BY created_at ASC").fetchall()
    # Compute points dynamically: posts*50 + comments*10
    result = []
    for r in rows:
        d = dict(r)
        post_count    = db.execute("SELECT COUNT(*) FROM posts    WHERE author_id=? AND is_deleted=0", (d["id"],)).fetchone()[0]
        comment_count = db.execute("SELECT COUNT(*) FROM comments WHERE author_id=?",               (d["id"],)).fetchone()[0]
        vote_row      = db.execute(
            "SELECT COALESCE(SUM(cv.vote),0) as vs FROM comment_votes cv "
            "JOIN comments c ON c.id=cv.comment_id WHERE c.author_id=?", (d["id"],)
        ).fetchone()
        d["points"]    = post_count * 50 + comment_count * 10
        d["vote_score"] = int(vote_row["vs"]) if vote_row else 0
        result.append(d)
    db.close()
    return result

@app.patch("/admin/profiles/{user_id}")
async def admin_update_profile(user_id: str, request: Request, user=Depends(require_admin)):
    data   = await request.json()
    db     = get_db()
    fields = {k:v for k,v in data.items() if k in ("role","is_banned","points")}
    if fields:
        sets = ", ".join(f"{k}=?" for k in fields)
        db.execute(f"UPDATE profiles SET {sets} WHERE id=?", (*fields.values(), user_id))
        # If unbanning, delete all their alerts so count resets to 0
        if fields.get("is_banned") == 0:
            db.execute("DELETE FROM alerts WHERE user_id=?", (user_id,))
        db.commit()
        # Push ban event to user's SSE channel so they are immediately signed out
        if fields.get("is_banned") == 1:
            target = db.execute("SELECT uid FROM profiles WHERE id=?", (user_id,)).fetchone()
            if target:
                broker.publish(f"user:{target['uid']}", {"type": "banned"})
    db.close()
    return {"ok": True}

@app.post("/admin/verify-id")
async def verify_id_and_grant(
    file: UploadFile = File(...),
    uid:  str        = Form(""),
    user=Depends(require_user),
):
    """
    One-shot ID verification: receive image, run Florence OCR, grant admin if valid.
    No URL fetching needed — image bytes arrive directly.
    """
    img_bytes = await file.read()
    db        = get_db()
    try:
        id_result = _siglip_check_id(img_bytes)
        print(f"[verify-id] uid={uid} is_id={id_result.get('is_id')} conf={id_result.get('confidence',0):.2f} ocr={repr(id_result.get('ocr','')[:80])}")

        if not id_result.get("is_id"):
            # Save image and create pending request right here — no second round trip
            ext      = file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "jpg"
            fname    = f"id_{uuid.uuid4().hex[:8]}.{ext}"
            img_path = os.path.join(IMG_DIR, fname)
            with open(img_path, "wb") as fh:
                fh.write(img_bytes)
            id_image_url = f"/images/{fname}"

            lookup_uid = uid.strip().lower() or user.get("uid", "")
            profile    = db.execute("SELECT id FROM profiles WHERE uid=?", (lookup_uid,)).fetchone()
            user_id    = profile["id"] if profile else str(uuid.uuid4())

            db.execute(
                "INSERT INTO admin_requests (id,user_id,email,name,role_title,reason,id_image_url) VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), user_id, lookup_uid, lookup_uid, "staff",
                 "Staff ID submitted — awaiting manual review", id_image_url)
            )
            db.commit()
            broker.publish("admin", {"type": "new_request"})
            print(f"[verify-id] pending request saved with image {id_image_url}")
            return {"auto_approved": False, "confidence": id_result.get("confidence", 0), "id_image_url": id_image_url, "pending_saved": True}

        # Save the ID image for audit trail
        ext      = file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "jpg"
        fname    = f"id_{uuid.uuid4().hex[:8]}.{ext}"
        img_path = os.path.join(IMG_DIR, fname)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        id_image_url = f"/images/{fname}"

        # Grant admin role
        lookup_uid = uid.strip().lower() or user.get("uid", "")
        profile    = db.execute("SELECT id FROM profiles WHERE uid=?", (lookup_uid,)).fetchone()
        if profile:
            db.execute("UPDATE profiles SET role='admin' WHERE id=?", (profile["id"],))

        # Log the approved request
        db.execute(
            "INSERT INTO admin_requests (id,user_id,email,name,role_title,reason,id_image_url,status) VALUES (?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), profile["id"] if profile else lookup_uid,
             lookup_uid, lookup_uid, "staff", "Auto-approved via ID scan", id_image_url, "approved")
        )
        db.commit()
        print(f"[verify-id] ✅ admin granted to uid={lookup_uid}")
        return {"auto_approved": True, "confidence": id_result.get("confidence", 0)}
    except Exception as e:
        print(f"[verify-id] error: {e}")
        return {"auto_approved": False, "error": str(e)}
    finally:
        db.close()

@app.post("/auth/verify-badge")
async def verify_badge(
    file: UploadFile = File(...),
    user=Depends(require_user),
):
    """
    Let a logged-in user upload their university ID to get a student/staff badge.
    Runs the ID check in a background thread and pushes the result via SSE so
    the response returns immediately without blocking the event loop.
    """
    img_bytes = await file.read()
    if not img_bytes:
        return {"ok": False, "badge": "none", "message": "No image received."}

    def _run(user_id: str, user_uid: str, image_bytes: bytes):
        try:
            result = _siglip_check_id(image_bytes)
            print(f"[verify-badge] uid={user_uid} is_id={result.get('is_id')} role={result.get('detected_role')}")
            if not result.get("is_id"):
                broker.publish(f"user:{user_uid}", {
                    "type":    "badge_result",
                    "ok":      False,
                    "badge":   "none",
                    "message": "Could not verify this as a university ID. Make sure the card is clearly visible.",
                })
                return
            detected = result.get("detected_role", "unknown")
            badge    = detected if detected in ("student", "staff") else "verified"
            db2 = get_db()
            db2.execute("UPDATE profiles SET badge=? WHERE id=?", (badge, user_id))
            db2.commit(); db2.close()
            print(f"[verify-badge] ✅ badge={badge} → uid={user_uid}")
            broker.publish(f"user:{user_uid}", {
                "type":    "badge_result",
                "ok":      True,
                "badge":   badge,
                "message": f"{'Student' if badge == 'student' else 'Staff' if badge == 'staff' else 'Verified'} badge added!",
            })
        except Exception as e:
            print(f"[verify-badge] error: {e}")
            broker.publish(f"user:{user_uid}", {
                "type":    "badge_result",
                "ok":      False,
                "badge":   "none",
                "message": "Verification failed. Please try again.",
            })

    threading.Thread(target=_run, args=(user["id"], user["uid"], img_bytes), daemon=True).start()
    return {"ok": "pending", "message": "Verifying your ID…"}

@app.post("/admin/requests")
async def submit_admin_request(request: Request):
    """Save a pending admin request. ID check already done by /admin/verify-id — just store it."""
    data    = await request.json()
    db      = get_db()
    uid     = data.get("uid", "").strip().lower()
    profile = db.execute("SELECT id FROM profiles WHERE uid=?", (uid,)).fetchone() if uid else None
    user_id = profile["id"] if profile else str(uuid.uuid4())

    id_image_url = data.get("id_image_url") or None

    db.execute(
        "INSERT INTO admin_requests (id,user_id,email,name,role_title,reason,id_image_url) VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), user_id, data.get("email", ""), data.get("name", ""),
         data.get("role_title", ""), data.get("reason", ""), id_image_url)
    )
    db.commit(); db.close()
    broker.publish("admin", {"type": "new_request"})
    return {"ok": True, "auto_approved": False, "message": "Request submitted — pending review"}

# ══════════════════════════════════════════════════════════════════════════════
# ── AI IMAGE FEATURES ─────────────────────────────────────────────────────────
#   Feature 1: POST /search/image      → image search in search bar
#   Feature 2: POST /posts/with-image  → create post + auto-match via SSE
#   Feature 3: NSFW check on upload    → POST /upload/checked
# ══════════════════════════════════════════════════════════════════════════════
import io, threading
import numpy as np
from PIL import Image

# ── lazy model singletons ─────────────────────────────────────────────────────
_dino_proc  = _dino_model  = None
_nsfw_proc  = _nsfw_model  = None
_model_lock        = threading.Lock()
_florence_lock     = threading.Lock()
_qwen_lock         = threading.Lock()
_siglip_lock       = threading.Lock()

def _load_dino():
    global _dino_proc, _dino_model
    if _dino_model is None:
        with _model_lock:
            if _dino_model is None:
                from transformers import AutoImageProcessor, AutoModel
                _dino_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
                _dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
                _dino_model.eval()
    return _dino_proc, _dino_model

def _load_nsfw():
    global _nsfw_proc, _nsfw_model
    if _nsfw_model is None:
        with _model_lock:
            if _nsfw_model is None:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                _nsfw_proc  = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
                _nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
                _nsfw_model.eval()
    return _nsfw_proc, _nsfw_model

# ── helpers ───────────────────────────────────────────────────────────────────
def _embed(img_bytes: bytes) -> list:
    import torch
    proc, model = _load_dino()
    img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().numpy().tolist()

def _embed_path(path: str) -> list:
    with open(path, "rb") as f:
        return _embed(f.read())

def _cosine(a, b) -> float:
    va, vb = np.array(a), np.array(b)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0

def _is_nsfw(img_bytes: bytes) -> bool:
    import torch
    proc, model = _load_nsfw()
    img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    label = model.config.id2label[int(logits.argmax(-1))]
    return label.lower() == "nsfw"

# ── DB migration: add embedding column ────────────────────────────────────────
def _migrate():
    db   = get_db()
    cols = [r[1] for r in db.execute("PRAGMA table_info(posts)").fetchall()]
    if "embedding" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN embedding TEXT")
        db.commit()
    db.close()
_migrate()

# ── startup backfill (embed existing posts that have images) ──────────────────
def _backfill():
    db   = get_db()
    rows = db.execute(
        "SELECT id, image_url FROM posts WHERE image_url IS NOT NULL AND embedding IS NULL AND is_deleted=0"
    ).fetchall()
    db.close()
    for row in rows:
        # strip leading /images/ to get filename
        fname = row["image_url"].lstrip("/images/").lstrip("/")
        path  = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            vec = _embed_path(path)
            db2 = get_db()
            db2.execute("UPDATE posts SET embedding=? WHERE id=?", (json.dumps(vec), row["id"]))
            db2.commit()
            db2.close()
        except Exception as e:
            print(f"[backfill] {row['id']}: {e}")

threading.Thread(target=_backfill, daemon=True).start()

# pending embeddings keyed by image url (temp store between upload & post save)
_pending_emb:    dict[str, list] = {}
_pending_siglip: dict[str, list] = {}
_pending_lock = threading.Lock()  # protect concurrent pop/set operations

# ── FEATURE 3: moderated image upload ─────────────────────────────────────────
@app.post("/upload/checked")
async def upload_checked(file: UploadFile = File(...), user=Depends(require_user)):
    """Upload image with NSFW check + DINOv2 embedding (pre-computed before post save)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()

    # ── NSFW gate ─────────────────────────────────────────────────────────────
    try:
        if _is_nsfw(data):
            raise HTTPException(422, "Image rejected: inappropriate content detected")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[nsfw] {e}")   # don't block upload on model error

    # ── save ─────────────────────────────────────────────────────────────────
    ext      = file.content_type.split("/")[1].replace("jpeg", "jpg")
    filename = f"{uuid.uuid4()}.{ext}"
    fpath    = os.path.join(IMG_DIR, filename)
    with open(fpath, "wb") as f:
        f.write(data)
    url = f"/images/{filename}"

    # ── embed synchronously — must be ready before /posts/ai is called ─────────
    # The user is already waiting for the upload response; 150-200ms extra is fine.
    try:
        _pending_emb[url] = _embed_path(fpath)
    except Exception as e:
        print(f"[embed upload] {e}")  # non-fatal: post saves without embedding

    return {"url": url}


# ── Module-level auto-match worker (avoids closure capture bug) ──────────────
def _run_auto_match(post_id: str, emb_json, siglip_json, opp: str, author_uid: str):
    """Run in a thread. All args passed by value so concurrent calls are safe.

    Scoring strategy:
    - SigLIP image↔image is semantically aware — threshold 0.25 catches genuine
      visual similarity (same object type/color) while rejecting unrelated images.
    - DINOv2 is a patch-level feature extractor, not semantic — it can score high
      for a car vs an ID card if they share background/lighting. Only use it as a
      fallback when a post has no SigLIP embedding yet (uploaded before backfill),
      and apply a stricter threshold of 0.50 to avoid false positives.
    - Never mix models: if SigLIP embedding exists on either side, use SigLIP only.
    """
    try:
        qvec_dino   = json.loads(emb_json)    if emb_json    else None
        qvec_siglip = json.loads(siglip_json) if siglip_json else None
        db2  = get_db()
        rows = db2.execute(
            "SELECT id, title, image_url, embedding, siglip_embedding FROM posts "
            "WHERE status=? AND is_deleted=0 AND id!=? "
            "AND (embedding IS NOT NULL OR siglip_embedding IS NOT NULL)",
            (opp, post_id)
        ).fetchall()
        db2.close()
        scored = []
        for r in rows:
            try:
                sim       = 0.0
                threshold = 1.0  # default — won't match unless set below

                if qvec_siglip and r["siglip_embedding"]:
                    # SigLIP↔SigLIP — semantically aware
                    sim       = _cosine(qvec_siglip, json.loads(r["siglip_embedding"]))
                    threshold = 0.50  # auto-notify: only fire for strong visual matches
                elif qvec_dino and r["embedding"] and not r["siglip_embedding"]:
                    # DINOv2 fallback for old posts without SigLIP embedding
                    sim       = _cosine(qvec_dino, json.loads(r["embedding"]))
                    threshold = 0.65  # DINOv2 is not semantic — need very high bar

                if sim >= threshold:
                    scored.append({"id": r["id"], "title": r["title"],
                                   "image_url": r["image_url"], "score": round(sim, 3)})
            except Exception:
                pass
        scored.sort(key=lambda x: x["score"], reverse=True)
        if scored:
            broker.publish(f"user:{author_uid}", {
                "type": "image_matches", "post_id": post_id, "matches": scored
            })
            print(f"[auto-match] {len(scored)} matches for post {post_id}")
        else:
            print(f"[auto-match] no matches for post {post_id}")
    except Exception as e:
        print(f"[auto-match] {e}")


# ── FEATURE 2: create post + auto-match ───────────────────────────────────────
@app.post("/posts/ai")
async def create_post_ai(request: Request, user=Depends(require_user)):
    """
    Same as POST /posts but:
    - Saves pre-computed embedding from _pending_emb
    - Triggers auto-match SSE for Lost↔Found after save
    """
    data  = await request.json()
    pid   = str(uuid.uuid4())
    iurl   = data.get("image_url")
    with _pending_lock:
        emb    = json.dumps(_pending_emb.pop(iurl))    if iurl and iurl in _pending_emb    else None
        siglip = json.dumps(_pending_siglip.pop(iurl)) if iurl and iurl in _pending_siglip else None

    # Generate AI caption WITHOUT holding a DB connection — Florence is slow
    # and SQLite WAL mode still blocks concurrent writers if a reader holds open.
    ai_caption = None
    if iurl:
        fname = iurl.lstrip("/images/").lstrip("/")
        fpath = os.path.join(IMG_DIR, fname)
        if os.path.exists(fpath):
            try:
                ai_caption = _get_image_caption(open(fpath, "rb").read())
                print(f"[post caption] {ai_caption[:60] if ai_caption else ''}")
            except Exception as _ce:
                print(f"[post caption error] {_ce}")

    # Open DB only now — after all slow work is done
    db = get_db()
    db.execute(
        "INSERT INTO posts (id,author_id,title,description,location,category,status,image_url,ai_caption,embedding,siglip_embedding) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (pid, user["id"], data["title"], data.get("description",""),
         data["location"], data["category"], data.get("status","found"), iurl, ai_caption, emb, siglip)
    )
    db.commit()
    row = db.execute("""
        SELECT p.*, pr.uid as author_uid, pr.name as author_name,
               pr.initials as author_initials, pr.color as author_color, 0 as comment_count
        FROM posts p JOIN profiles pr ON pr.id=p.author_id WHERE p.id=?
    """, (pid,)).fetchone()
    db.close()
    post_data = dict(row)
    broker.publish("all", {"type": "new_post", "post": post_data})

    # ── auto-match: search opposite-status posts by image ────────────────────
    status = data.get("status","found")
    if (emb or siglip) and status in ("lost", "found"):
        opposite = "found" if status == "lost" else "lost"

        threading.Thread(
            target=_run_auto_match,
            args=(pid, emb, siglip, opposite, user["uid"]),
            daemon=True
        ).start()

    return post_data


# ── FEATURE 1: image search endpoint ──────────────────────────────────────────
@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    status_filter: str = "all"
):
    """Search posts by image similarity. status_filter: all|lost|found|waiting|recovered"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        qvec = _embed(await file.read())
    except Exception as e:
        raise HTTPException(500, f"Could not process image: {e}")

    db  = get_db()
    sql = (
        "SELECT p.*, pr.uid as author_uid, pr.name as author_name, "
        "pr.initials as author_initials, pr.color as author_color, 0 as comment_count "
        "FROM posts p JOIN profiles pr ON pr.id=p.author_id "
        "WHERE p.is_deleted=0 AND p.embedding IS NOT NULL"
    )
    params = []
    if status_filter != "all":
        sql += " AND p.status=?"; params.append(status_filter)
    rows = db.execute(sql, params).fetchall()
    db.close()

    scored = []
    for r in rows:
        try:
            sim = _cosine(qvec, json.loads(r["embedding"]))
            if sim >= 0.20:  # 20% resemblance threshold
                d = dict(r); d["similarity"] = round(sim,3); d.pop("embedding",None)
                scored.append(d)
        except Exception:
            pass

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored  # no cap — return all matches above threshold

# ── DEBUG: check embedding status ─────────────────────────────────────────────
@app.get("/debug/embeddings")
async def debug_embeddings():
    """Shows which posts have embeddings stored. Remove this endpoint in production."""
    db   = get_db()
    rows = db.execute(
        "SELECT id, title, status, image_url, "
        "CASE WHEN embedding IS NULL THEN 0 ELSE 1 END as has_embedding "
        "FROM posts WHERE is_deleted=0 ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    db.close()
    total     = len(rows)
    with_emb  = sum(1 for r in rows if r["has_embedding"])
    return {
        "total_posts": total,
        "posts_with_embedding": with_emb,
        "posts_without_embedding": total - with_emb,
        "posts": [dict(r) for r in rows],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW AI FEATURES ───────────────────────────────────────────────────────────
#   F-A: Florence-2  → photo auto-fill (title, desc, category)
#   F-B: Qwen-0.5B   → natural language search parsing
#   F-C: SigLIP2     → live camera search + admin ID check
#   F-D: DINOv3      → upgraded image similarity (replaces DINOv2 gradually)
#   F-E: Cron        → auto-status nudge for stale posts
# ══════════════════════════════════════════════════════════════════════════════

# ── lazy model singletons (new) ───────────────────────────────────────────────
_florence_proc  = _florence_model  = None
_qwen_tok       = _qwen_model      = None
_siglip_proc    = _siglip_model    = None

def _load_florence():
    global _florence_proc, _florence_model
    if _florence_model is None:
        with _florence_lock:
            if _florence_model is None:
                import sys, types, torch
                from transformers import AutoProcessor, AutoModelForCausalLM

                # Florence-2's modeling file calls is_flash_attn_2_available()
                # which uses importlib.util.find_spec — stub needs __spec__ set
                if "flash_attn" not in sys.modules:
                    import importlib.util
                    stub = types.ModuleType("flash_attn")
                    stub.__spec__ = importlib.util.spec_from_loader("flash_attn", loader=None)
                    stub.__version__ = "0.0.0"
                    stub.flash_attn_func = None
                    stub.flash_attn_varlen_func = None
                    sys.modules["flash_attn"] = stub
                    sub = types.ModuleType("flash_attn.flash_attn_interface")
                    sub.__spec__ = importlib.util.spec_from_loader("flash_attn.flash_attn_interface", loader=None)
                    sys.modules["flash_attn.flash_attn_interface"] = sub

                _florence_proc  = AutoProcessor.from_pretrained(
                    "microsoft/Florence-2-base", trust_remote_code=True)
                _florence_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Florence-2-base", trust_remote_code=True,
                    attn_implementation="eager",
                    torch_dtype=torch.float32,
                )
                _florence_model.eval()
    return _florence_proc, _florence_model

def _load_qwen():
    global _qwen_tok, _qwen_model
    if _qwen_model is None:
        with _qwen_lock:
            if _qwen_model is None:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                _qwen_tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
                _qwen_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-0.5B-Instruct")
                _qwen_model.eval()
    return _qwen_tok, _qwen_model

def _load_siglip():
    global _siglip_proc, _siglip_model
    if _siglip_model is None:
        with _siglip_lock:
            if _siglip_model is None:
                from transformers import AutoModel, CLIPImageProcessor
                # Use CLIPImageProcessor (image-only, no sentencepiece dependency)
                # instead of AutoProcessor which also loads SiglipTokenizer
                _siglip_proc  = CLIPImageProcessor.from_pretrained(
                    "google/siglip-base-patch16-224")
                _siglip_model = AutoModel.from_pretrained(
                    "google/siglip-base-patch16-224")
                _siglip_model.eval()
    return _siglip_proc, _siglip_model


# ── Florence-2 helpers ────────────────────────────────────────────────────────
def _get_image_caption(img_bytes: bytes) -> str:
    """Return a short plain-text caption of what's in the image using Florence-2.
    Used to make posts searchable by image content ('car', 'red bag', etc.)."""
    import torch
    try:
        proc, model = _load_florence()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((384, 384))  # small for speed
        inputs = proc(text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=60, num_beams=1, do_sample=False,
            )
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
    except Exception as e:
        print(f"[get_image_caption] {e}")
        return ""


def _florence_describe(img_bytes: bytes, status: str = "lost", location: str = "") -> dict:
    """Returns {title, description, category} from image using Florence-2."""
    import torch, re, random
    proc, model = _load_florence()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Step 1: get raw caption from Florence
    inputs = proc(text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=120, num_beams=3, do_sample=False
        )
    caption = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # Step 2: extract core object noun (strip scene-setting filler)
    clean = re.sub(
        r"(the image shows?|it appears?( to be)?|the background( is)?|"
        r"the overall mood|the lighting|this is a (photo|image|picture) of)\s*",
        "", caption, flags=re.IGNORECASE
    ).strip().lstrip(",. ")

    # Short object label for title (max 6 words, up to first comma/period)
    short = re.split(r"[,.]", clean)[0].strip()
    item  = " ".join(short.split()[:6]).capitalize()
    if not item:
        item = " ".join(caption.split()[:5]).capitalize()

    # Step 3: extract only truly useful detail — color + object, strip background/context noise
    # Remove background descriptions ("on a blue background", "parked on a race track", etc.)
    detail_clean = re.sub(
        r"\b(on a|on the|in a|in the|against a|against the|parked on|placed on|sitting on)\b.*",
        "", clean, flags=re.IGNORECASE
    ).strip().rstrip(".,")
    # Take first sentence only
    detail = re.split(r"\. ", detail_clean)[0].strip().rstrip(".")
    if len(detail) > 90:
        detail = " ".join(detail.split()[:15])
    # Make sure detail doesn't just repeat the item name
    if detail.lower().strip() == item.lower().strip():
        detail = ""

    # Step 4: build unique human-like description
    verb_lost  = (status == "lost") if status else True
    loc_phrase = f"at {location}" if location and location.lower() not in ("unknown", "other", "") else "on campus"
    item_lower = item.lower()

    if verb_lost:
        openers = [
            f"Hey everyone, I lost my {item_lower} {loc_phrase} and I'm really hoping someone found it.",
            f"I accidentally left my {item_lower} {loc_phrase} — has anyone seen it?",
            f"I can't find my {item_lower} anywhere, I think I lost it {loc_phrase}.",
            f"Help! I lost my {item_lower} {loc_phrase} and I really need it back.",
        ]
        closers = [
            "Please DM me if you've found it, I'd really appreciate it!",
            "If you found it, please reach out — it means a lot to me.",
            "Any help finding it would be amazing, thank you!",
            "Please contact me if you have it, I'll come pick it up right away!",
        ]
    else:
        openers = [
            f"Hi! I found a {item_lower} {loc_phrase} — does it belong to anyone here?",
            f"Found a {item_lower} {loc_phrase}. Is anyone missing this?",
            f"Someone left their {item_lower} {loc_phrase}, I have it safe with me.",
            f"I picked up a {item_lower} {loc_phrase} — let me know if it's yours!",
        ]
        closers = [
            "Just describe it to me and I'll return it to you!",
            "Message me with a description and I'll get it back to you.",
            "Reach out if it's yours and I'll arrange to return it.",
            "Contact me with some details about it and I'll give it back!",
        ]

    opener = random.choice(openers)
    closer = random.choice(closers)

    if detail:
        description = f"{opener} It looks like: {detail.lower()}. {closer}"
    else:
        description = f"{opener} {closer}"

    title = item

    # Step 5: category from caption keywords — "Other" when nothing matches
    kw_map = {
        r"phone|laptop|tablet|charger|earphones?|headphones?|earbuds?|cable|usb|computer|keyboard|mouse": "Electronics",
        r"bag|backpack|purse|wallet|suitcase|pouch|handbag":                                               "Bags",
        r"watch|ring|necklace|bracelet|glasses|sunglasses|jewel":                                          "Accessories",
        r"\bhat\b|\bcap\b|\bjacket\b|\bshirt\b|\bpants\b|\bcoat\b|\bhoodie\b|\bshoe\b|\bboot\b|\bscarf\b|\bcloth": "Clothing",
        r"\bid\b|card|passport|license|badge|student":                                                     "ID / Cards",
        r"\bkey\b|keychain|keyring":                                                                       "Keys",
    }
    category  = "Other"
    cap_lower = caption.lower()
    for pattern, cat in kw_map.items():
        if re.search(pattern, cap_lower):
            category = cat; break

    return {"title": title, "description": description, "category": category}


# ── Qwen helpers ──────────────────────────────────────────────────────────────
def _qwen_parse_search(query: str) -> dict:
    """Parse natural language search query into structured filters."""
    import torch, re
    tok, model = _load_qwen()

    system = (
        "You are a search parser for a campus lost and found app. "
        "Extract search intent from the user query and return ONLY valid JSON with these fields: "
        '{"keywords": "main search terms", "status": "lost|found|waiting|recovered|all", '
        '"location": "location name or empty string", "category": "Electronics|Bags|Accessories|Clothing|ID / Cards|Keys|Other|empty string"}. '
        "No explanation, no markdown, only the JSON object."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": query}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=120,
            pad_token_id=tok.eos_token_id, do_sample=False
        )
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # extract JSON safely
    try:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        return json.loads(m.group()) if m else {"keywords": query, "status": "all", "location": "", "category": ""}
    except Exception:
        return {"keywords": query, "status": "all", "location": "", "category": ""}


# ── SigLIP2 helpers ───────────────────────────────────────────────────────────
_siglip_text_proc = None
_siglip_text_lock = threading.Lock()

def _load_siglip_text():
    """Load only the SigLIP text tokenizer — requires sentencepiece."""
    global _siglip_text_proc
    if _siglip_text_proc is None:
        with _siglip_text_lock:
            if _siglip_text_proc is None:
                from transformers import AutoTokenizer
                _siglip_text_proc = AutoTokenizer.from_pretrained(
                    "google/siglip-base-patch16-224")
    return _siglip_text_proc

def _siglip_embed_image(img_bytes: bytes) -> list:
    """Image embedding via SigLIP — uses CLIPImageProcessor, no sentencepiece needed."""
    import torch
    proc, model = _load_siglip()
    img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().tolist()

def _siglip_embed_text(text: str) -> list:
    """Text embedding via SigLIP — requires sentencepiece for tokenization."""
    import torch
    tok   = _load_siglip_text()
    _, model = _load_siglip()
    inputs = tok(text=[text], return_tensors="pt", padding="max_length",
                 truncation=True, max_length=64)
    with torch.no_grad():
        feats = model.get_text_features(input_ids=inputs["input_ids"])
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().tolist()

def _siglip_check_id(img_bytes: bytes) -> dict:
    """
    Detect if an image is a university/institution ID card.
    Also detects whether it's a student or staff/faculty card.
    Strategy: OCR the image with Florence-2, then look for ID-card keywords.
    """
    import re, torch

    # ── 1. OCR with Florence-2 ────────────────────────────────────────────
    ocr_text = ""
    try:
        proc, model = _load_florence()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Scale up small images so Florence OCR has enough resolution
        w, h = img.size
        if min(w, h) < 512:
            scale = 512 / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.thumbnail((1024, 1024), Image.LANCZOS)

        inputs = proc(text="<OCR>", images=img, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=300, num_beams=3, do_sample=False,
            )
        # Use post_process_generation to correctly strip Florence location tokens.
        # batch_decode with skip_special_tokens=True does NOT remove <loc_XYZ> tokens
        # because they are regular vocab tokens, not registered special tokens —
        # leaving "<loc_23><loc_45>student id<loc_87>" which breaks keyword matching.
        raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
        try:
            result = proc.post_process_generation(raw, task="<OCR>", image_size=(img.width, img.height))
            if isinstance(result, dict):
                ocr_text = " ".join(str(v) for v in result.values())
            else:
                ocr_text = str(result)
        except Exception:
            # Fallback: strip <loc_NNN> tokens with regex
            ocr_text = re.sub(r'<loc_\d+>', ' ', raw)
            ocr_text = re.sub(r'<[^>]+>', ' ', ocr_text)
        ocr_text = re.sub(r'\s+', ' ', ocr_text).strip().lower()
        print(f"[id-check] OCR ({len(ocr_text)} chars): {repr(ocr_text[:160])}")
    except Exception as e:
        print(f"[id-check] OCR failed: {e}")

    # ── 2. Keyword scoring ────────────────────────────────────────────────
    strong_keywords = [
        "student id", "staff id", "employee id", "faculty id",
        "university", "institute of technology", "college", "école",
        "student card", "id card", "identity card", "access card",
        "student", "matricule", "carte étudiant", "carte d'étudiant",
        "mit id", "campus card",
    ]
    weak_keywords = [
        "id", "name", "department", "valid", "expires", "issued",
        "badge", "member", "card no", "card #",
    ]

    strong_hits = [kw for kw in strong_keywords if kw in ocr_text]
    weak_hits   = [kw for kw in weak_keywords   if kw in ocr_text]

    # ── 3. Aspect ratio check ─────────────────────────────────────────────
    try:
        img_check = Image.open(io.BytesIO(img_bytes))
        w, h = img_check.size
        ratio = max(w, h) / max(min(w, h), 1)
        card_shape = 1.3 <= ratio <= 2.0
    except Exception:
        card_shape = False

    if strong_hits:
        confidence = min(0.95, 0.60 + len(strong_hits) * 0.10)
    elif len(weak_hits) >= 2 and card_shape:
        confidence = 0.55
    else:
        confidence = 0.10

    is_id = confidence >= 0.45

    # ── 4. Detect student vs staff from OCR text ──────────────────────────
    staff_keywords   = ["staff", "faculty", "employee", "professor", "instructor",
                        "lecturer", "researcher", "dr.", "prof.", "administrator",
                        "personnel", "enseignant", "professeur"]
    student_keywords = ["student", "étudiant", "undergraduate", "postgraduate",
                        "graduate", "élève", "stagiaire", "matricule"]

    detected_role = "unknown"
    if is_id:
        staff_hits   = [kw for kw in staff_keywords   if kw in ocr_text]
        student_hits = [kw for kw in student_keywords if kw in ocr_text]
        if staff_hits and not student_hits:
            detected_role = "staff"
        elif student_hits:
            detected_role = "student"
        elif staff_hits:
            detected_role = "staff"

    print(f"[id-check] strong={strong_hits} weak={weak_hits} card_shape={card_shape} → confidence={confidence:.2f} is_id={is_id} role={detected_role}")
    return {"is_id": is_id, "confidence": round(confidence, 3), "ocr": ocr_text[:200], "detected_role": detected_role}


# ── DB migration: add nudged_at column ────────────────────────────────────────
def _migrate_nudge():
    db   = get_db()
    cols = [r[1] for r in db.execute("PRAGMA table_info(posts)").fetchall()]
    if "nudged_at" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN nudged_at TEXT")
        db.commit()
    if "ai_caption" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN ai_caption TEXT")
        db.commit()
    db.close()
_migrate_nudge()

def _migrate_badge():
    db   = get_db()
    cols = [r[1] for r in db.execute("PRAGMA table_info(profiles)").fetchall()]
    if "badge" not in cols:
        db.execute("ALTER TABLE profiles ADD COLUMN badge TEXT NOT NULL DEFAULT 'none'")
        db.commit()
    db.close()
_migrate_badge()

def _migrate_reports_unique():
    """
    The old UNIQUE(post_id, reporter_id) meant reporting a post and then
    reporting a comment on that same post silently dropped the comment report
    (INSERT OR IGNORE hit the constraint). Fix: UNIQUE(post_id, reporter_id, reason)
    — post reports use a plain reason string, comment reports use 'comment:<id>:<reason>',
    so they're always distinct rows. Runs as rename+recreate since SQLite can't
    ALTER a UNIQUE constraint.
    """
    db = get_db()
    try:
        # Detect old constraint: try inserting two rows with same post_id/reporter_id
        # but different reason inside a savepoint — if second insert fails, migrate.
        test_post = db.execute("SELECT id, author_id FROM posts LIMIT 1").fetchone()
        if not test_post:
            db.close(); return  # fresh DB uses new schema already
        needs = False
        try:
            db.execute("SAVEPOINT _rmu")
            db.execute("INSERT INTO reports (id,post_id,reporter_id,reason) VALUES (?,?,?,?)",
                       ("__t1__", test_post["id"], test_post["author_id"], "__test_post__"))
            db.execute("INSERT INTO reports (id,post_id,reporter_id,reason) VALUES (?,?,?,?)",
                       ("__t2__", test_post["id"], test_post["author_id"], "comment:x:__test__"))
            db.execute("ROLLBACK TO SAVEPOINT _rmu")
        except Exception:
            db.execute("ROLLBACK TO SAVEPOINT _rmu")
            needs = True
        if needs:
            print("[migrate_reports] upgrading UNIQUE constraint…")
            db.executescript("""
                CREATE TABLE reports_new (
                    id          TEXT PRIMARY KEY,
                    post_id     TEXT NOT NULL REFERENCES posts(id),
                    reporter_id TEXT NOT NULL REFERENCES profiles(id),
                    reason      TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(post_id, reporter_id, reason)
                );
                INSERT OR IGNORE INTO reports_new SELECT * FROM reports;
                DROP TABLE reports;
                ALTER TABLE reports_new RENAME TO reports;
            """)
            print("[migrate_reports] done ✓")
    except Exception as e:
        print(f"[migrate_reports error] {e}")
    finally:
        try: db.close()
        except: pass

_migrate_reports_unique()

# ── F-E: auto-status cron (runs every 24h) ────────────────────────────────────
def _run_nudge_cron():
    while True:
        time.sleep(60 * 60 * 24)  # 24h
        try:
            db  = get_db()
            now = datetime.utcnow()
            # posts older than 14 days, not recovered, not nudged in last 7 days
            rows = db.execute("""
                SELECT p.id, p.title, pr.uid as author_uid
                FROM posts p JOIN profiles pr ON pr.id=p.author_id
                WHERE p.is_deleted=0
                  AND p.status != 'recovered'
                  AND datetime(p.created_at) < datetime('now', '-14 days')
                  AND (p.nudged_at IS NULL OR datetime(p.nudged_at) < datetime('now', '-7 days'))
            """).fetchall()
            for r in rows:
                broker.publish(f"user:{r['author_uid']}", {
                    "type": "nudge",
                    "post_id": r["id"],
                    "title":   r["title"],
                    "message": "Is this item still active? Tap to update or mark as recovered."
                })
                db.execute("UPDATE posts SET nudged_at=? WHERE id=?",
                           (now.isoformat(), r["id"]))
            db.commit()
            db.close()
            print(f"[nudge cron] nudged {len(rows)} posts")
        except Exception as e:
            print(f"[nudge cron error] {e}")

threading.Thread(target=_run_nudge_cron, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW ENDPOINTS ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── F-A: auto-fill from photo ─────────────────────────────────────────────────
@app.post("/ai/describe-image")
async def describe_image(
    file:     UploadFile = File(...),
    status:   str        = Form("lost"),
    location: str        = Form(""),
    user=Depends(get_current_user),
):
    """Upload a photo → get back {title, description, category} auto-filled by Florence-2."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        result = _florence_describe(await file.read(), status=status, location=location)
        return result
    except Exception as e:
        raise HTTPException(500, f"Could not describe image: {e}")

# ── F-B: natural language search ─────────────────────────────────────────────
def _rule_parse_search(q: str) -> dict:
    """Fast rule-based NL parser — no model needed, instant."""
    import re
    q_low = q.lower()

    # status
    status = "all"
    if re.search(r"\blost\b|\bperdu\b|\bمفقود\b", q_low):   status = "lost"
    elif re.search(r"\bfound\b|\btrouvé\b|\bوجد\b", q_low): status = "found"

    # category
    category = ""
    cat_map = {
        "Electronics": r"phone|laptop|tablet|charger|earphone|headphone|cable|usb|computer|téléphone|ordinateur",
        "Bags":        r"bag|backpack|purse|wallet|sac|cartable|حقيبة",
        "Accessories": r"watch|ring|glasses|sunglasses|bracelet|jewelry|montre|lunettes",
        "Clothing":    r"jacket|shirt|coat|hoodie|scarf|hat|cap|veste|manteau",
        "ID / Cards":  r"id|card|badge|carte|بطاقة",
        "Keys":        r"key|keychain|clé|مفتاح",
    }
    for cat, pattern in cat_map.items():
        if re.search(pattern, q_low):
            category = cat; break

    # location — extract word after location prepositions
    loc = ""
    loc_match = re.search(r"(?:near|at|in|beside|next to|devant|dans|عند|بجانب)\s+([\w\s]+?)(?:\s|$|,|\.)", q_low)
    if loc_match:
        loc = loc_match.group(1).strip()

    # strip status/category/location words from keywords
    keywords = q_low
    for pat in [r"\b(lost|found|perdu|trouvé)\b", r"\b(near|at|in beside|next to)\b"]:
        keywords = re.sub(pat, "", keywords)
    keywords = " ".join(keywords.split())

    return {"keywords": keywords, "status": status, "location": loc, "category": category}


@app.get("/ai/search")
async def ai_search(q: str = ""):
    """
    Hybrid search: SigLIP2 text→image semantic search + keyword fallback.
    Returns all results sorted by best score. Always includes keyword matches
    so queries like 'car' find posts even without perfect embeddings.
    """
    if not q.strip():
        return []

    q_clean = q.strip()
    q_lower = q_clean.lower()
    words   = [w for w in q_lower.split() if len(w) >= 2]

    # ── Try SigLIP semantic embedding ────────────────────────────────────────
    qvec = None
    try:
        qvec = _siglip_embed_text(q_clean)
    except Exception as e:
        print(f"[ai/search embed error] {e}")

    db   = get_db()
    rows = db.execute(
        "SELECT p.id, p.title, p.description, p.location, p.category, p.status, "
        "p.image_url, p.ai_caption, p.created_at, p.author_id, p.siglip_embedding, "
        "pr.uid as author_uid, pr.name as author_name, "
        "pr.initials as author_initials, pr.color as author_color, "
        "pr.role as author_role, pr.badge as author_badge, "
        "(SELECT COUNT(*) FROM comments c WHERE c.post_id=p.id) as comment_count "
        "FROM posts p JOIN profiles pr ON pr.id=p.author_id "
        "WHERE p.is_deleted=0"
    ).fetchall()
    db.close()

    results = {}  # id → {post_dict, similarity}

    for r in rows:
        d   = dict(r)
        pid = d["id"]
        emb_json = d.pop("siglip_embedding", None)

        # ── Keyword score (always computed) ──────────────────────────────────
        # Include ai_caption so "car" matches posts where Florence saw a car
        text = " ".join(filter(None, [
            d.get('title',''), d.get('description',''),
            d.get('category',''), d.get('location',''),
            d.get('ai_caption',''),   # ← Florence image caption for visual search
        ])).lower()
        # Full-word hits score 1.0, partial/substring hits score 0.5
        kw_hits = sum(1.0 if f' {w} ' in f' {text} ' else (0.5 if w in text else 0.0)
                      for w in words)
        kw_score  = kw_hits / max(len(words), 1) if kw_hits > 0 else 0.0

        # ── Semantic score (only if embedding available) ──────────────────────
        sem_score = 0.0
        if qvec and emb_json:
            try:
                sem_score = _cosine(qvec, json.loads(emb_json))
            except Exception:
                pass

        # ── Combine scores ────────────────────────────────────────────────────
        # kw_score  : 0.0–1.0  (fraction of query words found in text)
        # sem_score : 0.0–1.0  (raw cosine, SigLIP text→image typically 0.05–0.35)
        #
        # We want "20%" shown to the user to mean a genuine 20% cosine match.
        # So we keep sem_score raw and use kw_score as-is, take the max,
        # then filter at 0.20 for semantic and 0.0 for keyword (any kw hit shows).
        combined = max(kw_score, sem_score)

        # Include if keyword matched (any score) or semantic ≥ 20%
        if kw_score > 0 or sem_score >= 0.20:
            d["similarity"] = round(combined, 3)
            results[pid] = d

    # Sort by combined score descending, return top 20
    sorted_results = sorted(results.values(), key=lambda x: x["similarity"], reverse=True)
    return sorted_results  # no cap — return all matches


# ── F-C: live camera search (SigLIP2 image-to-image) ─────────────────────────
@app.post("/ai/camera-search")
async def camera_search(
    file: UploadFile = File(...),
    status_filter: str = "all"
):
    """Fast image search using SigLIP2 — for live camera scanning."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        qvec = _siglip_embed_image(await file.read())
    except Exception as e:
        raise HTTPException(500, f"Could not process image: {e}")

    db  = get_db()
    sql = (
        "SELECT p.id, p.title, p.status, p.location, p.image_url, p.siglip_embedding "
        "FROM posts p WHERE p.is_deleted=0 AND p.siglip_embedding IS NOT NULL"
    )
    params = []
    if status_filter != "all":
        sql += " AND p.status=?"; params.append(status_filter)
    rows = db.execute(sql, params).fetchall()
    db.close()

    scored = []
    for r in rows:
        try:
            sim = _cosine(qvec, json.loads(r["siglip_embedding"]))
            if sim > 0.15:
                scored.append({
                    "id":         r["id"],
                    "title":      r["title"],
                    "status":     r["status"],
                    "location":   r["location"],
                    "image_url":  r["image_url"],
                    "similarity": round(sim, 3)
                })
        except Exception:
            pass

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:5]


# ── F-C: SigLIP migration + backfill ─────────────────────────────────────────
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"  # update this if model changes

def _migrate_siglip():
    db   = get_db()
    cols = [r[1] for r in db.execute("PRAGMA table_info(posts)").fetchall()]
    if "siglip_embedding" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN siglip_embedding TEXT")
        db.commit()

    # track which model generated the stored embeddings
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    stored = db.execute("SELECT value FROM meta WHERE key='siglip_model'").fetchone()
    if stored is None or stored[0] != SIGLIP_MODEL_ID:
        db.execute("UPDATE posts SET siglip_embedding=NULL")
        db.execute("INSERT OR REPLACE INTO meta (key,value) VALUES ('siglip_model',?)",
                   (SIGLIP_MODEL_ID,))
        db.commit()
        print(f"[siglip] model changed → wiped embeddings, will recompute")
    db.commit()
    db.close()

_migrate_siglip()

def _backfill_siglip():
    db   = get_db()
    rows = db.execute(
        "SELECT id, image_url FROM posts "
        "WHERE image_url IS NOT NULL AND siglip_embedding IS NULL AND is_deleted=0"
    ).fetchall()
    db.close()
    for row in rows:
        fname = row["image_url"].lstrip("/images/").lstrip("/")
        path  = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path): continue
        try:
            vec = _siglip_embed_image(open(path, "rb").read())
            db2 = get_db()
            db2.execute("UPDATE posts SET siglip_embedding=? WHERE id=?",
                        (json.dumps(vec), row["id"]))
            db2.commit(); db2.close()
        except Exception as e:
            print(f"[siglip backfill] {row['id']}: {e}")

threading.Thread(target=_backfill_siglip, daemon=True).start()


def _backfill_captions():
    """Generate Florence-2 captions for posts that have images but no ai_caption."""
    import time as _time
    _time.sleep(10)  # wait for SigLIP backfill to start first
    db   = get_db()
    rows = db.execute(
        "SELECT id, image_url FROM posts "
        "WHERE image_url IS NOT NULL AND ai_caption IS NULL AND is_deleted=0"
    ).fetchall()
    db.close()
    print(f"[caption backfill] {len(rows)} posts to caption")
    for row in rows:
        fname = row["image_url"].lstrip("/images/").lstrip("/")
        path  = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            caption = _get_image_caption(open(path, "rb").read())
            if caption:
                db2 = get_db()
                db2.execute("UPDATE posts SET ai_caption=? WHERE id=?",
                            (caption, row["id"]))
                db2.commit(); db2.close()
        except Exception as e:
            print(f"[caption backfill] {row['id']}: {e}")

threading.Thread(target=_backfill_captions, daemon=True).start()


# ── YOLO-World: fast open-vocabulary object detector ────────────────────────
# ~300-500ms on CPU vs 3-8s for OWLv2. Uses CNN not ViT — much faster.
# YOLOWorld takes a text prompt like "white headphones" → returns bounding boxes.
_yw_model = None
def _extract_yolo_query(text: str) -> str:
    """
    Extract the single best YOLO label from a natural language description.
    Strategy: find the first concrete object noun after 'a/an/the' or 'of'.
    YOLO works best with simple COCO-style labels: 'headphones', 'backpack', 'bottle'.
    """
    import re

    # Known COCO/YOLO object classes — prefer these if found anywhere in the text
    yolo_classes = [
        "headphones","earphones","earbuds","backpack","bag","wallet","purse","phone",
        "mobile","laptop","tablet","keyboard","mouse","monitor","charger","cable",
        "bottle","cup","mug","glass","book","notebook","pen","pencil","glasses",
        "sunglasses","hat","cap","helmet","jacket","shirt","shoe","watch","ring",
        "bracelet","necklace","keys","remote","umbrella","ball","toy","box","chair",
        "table","desk","sofa","couch","bed","door","window","bicycle","car","person",
        "cat","dog","bottle","scissors","knife","fork","spoon","bowl","plate",
    ]
    text_lower = text.lower()
    for cls in yolo_classes:
        if cls in text_lower:
            return cls

    # Fallback: first noun after 'a/an/the' at the start of the sentence
    m = re.search(r'\b(?:a|an|the)\s+(?:\w+\s+)?(\w{4,})\b', text_lower)
    if m:
        noise = {"pair","kind","type","sort","piece","set","lot","bit","group","bunch"}
        word = m.group(1)
        if word not in noise:
            return word

    # Last resort: first long word
    words = re.findall(r'\b\w{4,}\b', text_lower)
    noise = {"this","that","with","have","from","they","some","into","there",
             "image","shows","photo","picture","resting","wearing","holding"}
    for w in words:
        if w not in noise:
            return w

    return "object"

def _load_yolo_world():
    global _yw_model
    if _yw_model is None:
        print("[yolo-world] loading…")
        from ultralytics import YOLOWorld as _YW
        _yw_model = _YW("yolov8s-worldv2.pt")
        # Force CLIP download NOW at startup by setting a dummy class
        # This prevents the 338MB download from blocking the first real request
        _yw_model.set_classes(["object"])
        global _yw_current_classes
        _yw_current_classes = None  # reset so real query sets properly
        print("[yolo-world] ready ✓")
    return _yw_model

def _yolo_world_find(image: Image.Image, text_query: str, threshold: float = 0.05):
    global _yw_current_classes
    model = _load_yolo_world()
    # Only call set_classes if query changed — avoids re-downloading CLIP every frame
    if _yw_current_classes != text_query:
        model.set_classes([text_query])
        _yw_current_classes = text_query
    results = model.predict(image, conf=threshold, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "score": float(box.conf[0]),
                "box":   [float(x) for x in box.xyxy[0].tolist()],
            })
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections

_ref_image_query_cache: dict = {}   # md5 → {"query": str, "embedding": list}
_CARD_LIKE_NOUNS = {"card","id","badge","pass","ticket","document","license","permit","certificate"}

def _yolo_world_find_by_image(frame: Image.Image, query_img: Image.Image, threshold: float = 0.01):
    """
    Hybrid reference-image finder:
    1. Caption the ref image once with Florence (cached by md5).
    2. Extract core noun.
    3a. If noun is a flat/card-like item YOLO can't detect → SigLIP sliding-window similarity.
    3b. Otherwise → YOLO-World with the noun.
    """
    import torch, hashlib, io as _io, numpy as np

    buf = _io.BytesIO()
    query_img.save(buf, format="JPEG", quality=60)
    img_hash = hashlib.md5(buf.getvalue()).hexdigest()

    if img_hash not in _ref_image_query_cache:
        try:
            proc, model = _load_florence()
            q = query_img.copy(); q.thumbnail((384, 384))
            inputs = proc(text="<MORE_DETAILED_CAPTION>", images=q, return_tensors="pt")
            with torch.no_grad():
                ids = model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    max_new_tokens=80, num_beams=3, do_sample=False,
                )
            caption = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            query   = _extract_yolo_query(caption)
            print(f"[ref-image] '{caption}' → '{query}'")
        except Exception as e:
            print(f"[ref-image error] {e}")
            query = "object"

        # Pre-compute SigLIP image embedding of ref (used for card/flat-item sliding window)
        ref_buf = _io.BytesIO()
        query_img.save(ref_buf, format="JPEG")
        ref_emb = _siglip_embed_image(ref_buf.getvalue())
        _ref_image_query_cache[img_hash] = {"query": query, "embedding": ref_emb}

    cached    = _ref_image_query_cache[img_hash]
    query     = cached["query"]
    ref_emb   = cached["embedding"]

    # ── Card/flat items: SigLIP sliding window ─────────────────────────────
    if query in _CARD_LIKE_NOUNS or query == "object":
        W, H = frame.size
        best_score, best_box = 0.0, None
        # Try 3 scales × sliding windows
        for scale in [0.25, 0.40, 0.60]:
            ww, wh = max(60, int(W * scale)), max(40, int(H * scale))
            step_x, step_y = max(20, ww // 3), max(20, wh // 3)
            for x in range(0, W - ww + 1, step_x):
                for y in range(0, H - wh + 1, step_y):
                    patch = frame.crop((x, y, x + ww, y + wh))
                    pb    = _io.BytesIO(); patch.save(pb, format="JPEG", quality=70)
                    sim   = _cosine(ref_emb, _siglip_embed_image(pb.getvalue()))
                    if sim > best_score:
                        best_score, best_box = sim, [x, y, x + ww, y + wh]
        print(f"[ref-image sliding] best_sim={round(best_score,3)}")
        if best_score > 0.70 and best_box:
            return [{"score": float(best_score), "box": best_box}]
        return []

    # ── Normal objects: YOLO-World ──────────────────────────────────────────
    return _yolo_world_find(frame, query, threshold)


# ── F-C: store SigLIP embedding on upload ────────────────────────────────────
# Preload YOLO-World in background so first scan is instant
threading.Thread(target=_load_yolo_world, daemon=True).start()
# Patch upload/checked to also compute siglip embedding
_orig_upload_checked = upload_checked.__wrapped__ if hasattr(upload_checked, '__wrapped__') else None

@app.post("/upload/checked/v2")
async def upload_checked_v2(file: UploadFile = File(...), user=Depends(require_user)):
    """
    Full pipeline upload:
    1. NSFW check
    2. Save file
    3. Compute DINOv2 embedding (for similarity search)
    4. Compute SigLIP2 embedding (for camera search)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()

    # NSFW gate
    try:
        if _is_nsfw(data):
            raise HTTPException(422, "Image rejected: inappropriate content detected")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[nsfw] {e}")

    # save
    ext      = file.content_type.split("/")[1].replace("jpeg", "jpg")
    filename = f"{uuid.uuid4()}.{ext}"
    fpath    = os.path.join(IMG_DIR, filename)
    with open(fpath, "wb") as f:
        f.write(data)
    url = f"/images/{filename}"

    # DINOv2 embedding
    try:
        _emb = _embed_path(fpath)
        with _pending_lock:
            _pending_emb[url] = _emb
    except Exception as e:
        print(f"[dino embed] {e}")

    # SigLIP2 embedding
    try:
        _siglip_emb = _siglip_embed_image(data)
        with _pending_lock:
            _pending_siglip[url] = _siglip_emb
    except Exception as e:
        print(f"[siglip embed] {e}")

    return {"url": url, "fullUrl": f"{os.environ.get('SPACE_URL', '')}{url}"}


# ── F-D: admin ID check ───────────────────────────────────────────────────────
@app.post("/ai/check-id")
async def check_id_image(file: UploadFile = File(...)):
    """Check if uploaded image looks like a staff/student ID card using SigLIP2."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        result = _siglip_check_id(await file.read())
        return result
    except Exception as e:
        raise HTTPException(500, f"Could not analyse image: {e}")


# ── /ai/find-in-frame — real-world object finder ─────────────────────────────
@app.websocket("/ws/camera-search")
async def camera_search_ws(websocket: WebSocket):
    import asyncio, base64
    from concurrent.futures import ThreadPoolExecutor
    _executor = ThreadPoolExecutor(max_workers=1)

    await websocket.accept()
    target  = None
    ref_img = None

    async def run_yolo(frame_img):
        """Run YOLO in a thread so it doesn't block the async event loop."""
        loop = asyncio.get_event_loop()
        if ref_img:
            return await loop.run_in_executor(_executor, _yolo_world_find_by_image, frame_img, ref_img)
        else:
            return await loop.run_in_executor(_executor, _yolo_world_find, frame_img, target)

    try:
        while True:
            data = await websocket.receive()

            # ── Config message ───────────────────────────────────────────────
            if "text" in data:
                msg     = json.loads(data["text"])
                target  = msg.get("target", "").strip()
                ref_b64 = msg.get("ref_image_b64")

                if ref_b64:
                    ref_img = Image.open(io.BytesIO(base64.b64decode(ref_b64))).convert("RGB")
                    target  = None
                else:
                    ref_img = None

                # Warm up in thread — sets classes + downloads CLIP if needed
                if target:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(_executor, _yolo_world_find,
                                               Image.new("RGB", (64, 64)), target)

                await websocket.send_json({"status": "ready"})
                continue

            # ── Frame bytes ──────────────────────────────────────────────────
            if "bytes" in data:
                if not target and not ref_img:
                    await websocket.send_json({"found": False, "box": None, "confidence": 0.0})
                    continue

                try:
                    frame_img = Image.open(io.BytesIO(data["bytes"])).convert("RGB")
                    W, H = frame_img.size
                    if W > 640:
                        s = 640 / W
                        frame_img = frame_img.resize((640, int(H * s)), Image.BILINEAR)
                        W, H = frame_img.size

                    detections = await run_yolo(frame_img)

                    if detections:
                        x1, y1, x2, y2 = detections[0]["box"]
                        await websocket.send_json({
                            "found":      True,
                            "box":        [x1/W, y1/H, x2/W, y2/H],
                            "confidence": round(detections[0]["score"], 2),
                        })
                    else:
                        await websocket.send_json({"found": False, "box": None, "confidence": 0.0})

                except Exception as e:
                    print(f"[ws frame] {e}")
                    await websocket.send_json({"found": False, "box": None, "confidence": 0.0})

    except Exception:
        _executor.shutdown(wait=False)

# Keep the old HTTP endpoint for backwards compat but just call yolo-world
@app.post("/ai/find-in-frame")
async def find_in_frame(
    frame:     UploadFile = File(...),
    ref_image: Optional[UploadFile] = File(None),
    target:    str = Form(""),
):
    frame_bytes = await frame.read()
    frame_img   = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    W, H        = frame_img.size
    if W > 640:
        scale = 640 / W
        frame_img = frame_img.resize((640, int(H * scale)), Image.BILINEAR)
        W, H = frame_img.size

    target_name = target.strip() if target and target != "__ref_image__" else "your item"
    try:
        if target and target != "__ref_image__":
            # Text query always takes priority — fastest and most accurate
            yolo_query = _extract_yolo_query(target)
            print(f"[camera] '{target}' → '{yolo_query}'")
            detections = _yolo_world_find(frame_img, yolo_query, threshold=0.01)
        elif ref_image:
            # Image-only mode: Florence captions the ref image → extract noun → YOLO
            import hashlib, io as _refio
            ref_bytes = await ref_image.read()
            ref_img   = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
            detections = _yolo_world_find_by_image(frame_img, ref_img)
            # Get the noun Florence derived (cached after first call) and return it
            # so the frontend can send it as plain text on subsequent frames,
            # skipping Florence entirely and matching the fast text-query path.
            buf = _refio.BytesIO(); ref_img.save(buf, format="JPEG", quality=60)
            yolo_query = (_ref_image_query_cache.get(hashlib.md5(buf.getvalue()).hexdigest()) or {}).get("query", "?")
        else:
            return {"found": False, "box": None, "label": "", "confidence": 0.0}
        print(f"[camera] {W}x{H} → {len(detections)} detections, top={round(detections[0]['score'],2) if detections else 'none'}")
    except Exception as e:
        print(f"[find-in-frame] {e}")
        return {"found": False, "box": None, "label": "", "confidence": 0.0}

    if not detections:
        return {"found": False, "box": None, "label": yolo_query, "confidence": 0.0}
    best = detections[0]
    x1, y1, x2, y2 = best["box"]
    return {"found": True, "box": [x1/W, y1/H, x2/W, y2/H], "label": yolo_query, "confidence": round(best["score"], 2)}


@app.post("/debug/yolo-world")
async def debug_yolo_world(file: UploadFile = File(...), target: str = Form("headphones")):
    """Test YOLO-World: upload any photo, specify what to find."""
    import traceback
    try:
        data       = await file.read()
        img        = Image.open(io.BytesIO(data)).convert("RGB")
        detections = _yolo_world_find(img, target, threshold=0.01)
        return {
            "target":          target,
            "image_size":      f"{img.width}x{img.height}",
            "detections":      detections[:5],
            "model_ready":     _yw_model is not None,
            "current_classes": _yw_current_classes,
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/debug/florence")
async def debug_florence():
    """Check if Florence-2 loads correctly. Remove in production."""
    import sys, traceback
    steps = []
    try:
        steps.append("importing torch")
        import torch
        steps.append(f"torch ok — version {torch.__version__}")

        steps.append("importing transformers AutoProcessor")
        from transformers import AutoProcessor
        steps.append("importing transformers AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM
        steps.append("transformers ok")

        steps.append("loading processor from microsoft/Florence-2-base")
        proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        steps.append("processor loaded ✓")

        steps.append("loading model from microsoft/Florence-2-base")
        import sys, types, importlib.util
        if "flash_attn" not in sys.modules:
            stub = types.ModuleType("flash_attn")
            stub.__spec__ = importlib.util.spec_from_loader("flash_attn", loader=None)
            stub.__version__ = "0.0.0"
            stub.flash_attn_func = None
            stub.flash_attn_varlen_func = None
            sys.modules["flash_attn"] = stub
            sub = types.ModuleType("flash_attn.flash_attn_interface")
            sub.__spec__ = importlib.util.spec_from_loader("flash_attn.flash_attn_interface", loader=None)
            sys.modules["flash_attn.flash_attn_interface"] = sub
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True,
            attn_implementation="eager", torch_dtype=torch.float32,
        )
        steps.append("model loaded ✓")

        steps.append("running test inference")
        from PIL import Image as PILImage
        import io as _io
        # tiny 32x32 white image
        img = PILImage.new("RGB", (32, 32), color=(255,255,255))
        buf = _io.BytesIO(); img.save(buf, format="JPEG"); buf.seek(0)
        inputs = proc(text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20, num_beams=1, do_sample=False
            )
        out = proc.batch_decode(ids, skip_special_tokens=True)[0]
        steps.append(f"inference ok — output: '{out}'")
        return {"ok": True, "steps": steps}
    except Exception as e:
        tb = traceback.format_exc()
        return {"ok": False, "steps": steps, "error": str(e), "traceback": tb}


# ── DEBUG: test NL search ─────────────────────────────────────────────────────
@app.get("/debug/search")
async def debug_search(q: str = "lost keys near library"):
    filters = _rule_parse_search(q)
    return {"query": q, "parsed": filters}


# ── DEBUG: test semantic search ───────────────────────────────────────────────
@app.get("/debug/semantic-search")
async def debug_semantic_search(q: str = "keys on a table"):
    import traceback
    try:
        qvec = _siglip_embed_text(q)
    except Exception as e:
        return {"ok": False, "error": f"embed failed: {e}", "traceback": traceback.format_exc()}

    db = get_db()
    rows = db.execute(
        "SELECT id, title, image_url, siglip_embedding "
        "FROM posts WHERE is_deleted=0 AND siglip_embedding IS NOT NULL"
    ).fetchall()
    db.close()

    scores = []
    for r in rows:
        try:
            sim = _cosine(qvec, json.loads(r["siglip_embedding"]))
            scores.append({"id": r["id"], "title": r["title"], "similarity": round(sim, 4)})
        except Exception as e:
            scores.append({"id": r["id"], "title": r["title"], "error": str(e)})

    scores.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    db2 = get_db()
    stored_model = db2.execute("SELECT value FROM meta WHERE key='siglip_model'").fetchone()
    db2.close()
    return {
        "ok": True,
        "query": q,
        "model_in_db": stored_model[0] if stored_model else "unknown",
        "model_loaded": SIGLIP_MODEL_ID,
        "text_vec_len": len(qvec),
        "scores": scores
    }


@app.get("/debug/reembed")
async def debug_reembed():
    """Wipe and recompute all siglip embeddings with current model. No auth — remove in prod."""
    global _siglip_proc, _siglip_model

    # Force unload cached model so it reloads with correct model id
    _siglip_proc  = None
    _siglip_model = None

    db = get_db()
    db.execute("UPDATE posts SET siglip_embedding=NULL")
    db.execute("INSERT OR REPLACE INTO meta (key,value) VALUES ('siglip_model','__reset__')")
    db.commit()
    db.close()
    _migrate_siglip()

    db   = get_db()
    rows = db.execute(
        "SELECT id, image_url FROM posts WHERE image_url IS NOT NULL AND is_deleted=0"
    ).fetchall()
    db.close()
    done, failed, skipped = 0, 0, 0
    for row in rows:
        fname = os.path.basename(row["image_url"])
        path  = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            skipped += 1; continue
        try:
            vec = _siglip_embed_image(open(path, "rb").read())
            db2 = get_db()
            db2.execute("UPDATE posts SET siglip_embedding=? WHERE id=?",
                        (json.dumps(vec), row["id"]))
            db2.commit(); db2.close()
            done += 1
        except Exception as e:
            print(f"[reembed] {row['id']}: {e}"); failed += 1
    return {"reembedded": done, "failed": failed, "skipped": skipped}


# ── ADMIN: re-embed all posts with current SigLIP model ──────────────────────
@app.post("/admin/reembed-siglip")
async def reembed_siglip(user=Depends(require_admin)):
    """
    Wipe all siglip_embedding values and regenerate them using the currently
    loaded SigLIP model. Run this whenever you switch SigLIP model versions.
    """
    db = get_db()
    rows = db.execute(
        "SELECT id, image_url FROM posts WHERE is_deleted=0 AND image_url IS NOT NULL"
    ).fetchall()
    db.close()

    done, failed = 0, 0
    for r in rows:
        try:
            img_path = r["image_url"]
            if not img_path.startswith("/"):
                img_path = IMG_DIR + "/" + img_path.split("/")[-1]
            if not os.path.exists(img_path):
                failed += 1; continue
            vec = _siglip_embed_image(open(img_path, "rb").read())
            db2 = get_db()
            db2.execute("UPDATE posts SET siglip_embedding=? WHERE id=?",
                        (json.dumps(vec), r["id"]))
            db2.commit(); db2.close()
            done += 1
        except Exception as e:
            print(f"[reembed] {r['id']} failed: {e}")
            failed += 1

    return {"reembedded": done, "failed": failed}


# ── DEBUG: show actual similarity scores ──────────────────────────────────────
@app.get("/debug/scores")
async def debug_scores(q: str = "keys on a table"):
    try:
        qvec = _siglip_embed_text(q)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    db   = get_db()
    rows = db.execute(
        "SELECT id, title, siglip_embedding FROM posts WHERE is_deleted=0 AND siglip_embedding IS NOT NULL"
    ).fetchall()
    db.close()
    scores = []
    for r in rows:
        try:
            sim = _cosine(qvec, json.loads(r["siglip_embedding"]))
            scores.append({"title": r["title"], "similarity": round(sim, 4)})
        except:
            pass
    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return {"ok": True, "query": q, "scores": scores}

@app.get("/debug/embeddings-check")
async def debug_embeddings_check():
    """Check what image paths exist vs what DB has."""
    db = get_db()
    rows = db.execute(
        "SELECT id, title, image_url, "
        "CASE WHEN siglip_embedding IS NULL THEN 0 ELSE 1 END as has_emb "
        "FROM posts WHERE is_deleted=0"
    ).fetchall()
    db.close()

    results = []
    for r in rows:
        url = r["image_url"] or ""
        # try all path variants
        fname    = os.path.basename(url)
        path1    = os.path.join(IMG_DIR, fname)
        path2    = url if url.startswith("/") else None
        exists1  = os.path.exists(path1)
        exists2  = os.path.exists(path2) if path2 else False
        results.append({
            "title":     r["title"],
            "image_url": url,
            "has_emb":   bool(r["has_emb"]),
            "path_tried": path1,
            "file_exists": exists1 or exists2,
        })

    img_files = os.listdir(IMG_DIR) if os.path.exists(IMG_DIR) else []
    return {
        "IMG_DIR": IMG_DIR,
        "files_in_dir": img_files[:20],
        "posts": results
    }


# ── DEBUG: test find-in-frame with a URL ─────────────────────────────────────
@app.get("/debug/find-in-frame")
async def debug_find_in_frame(target: str = "keys", img_url: str = ""):
    """Test the finder using an existing uploaded image URL."""
    import urllib.request, traceback
    try:
        if img_url.startswith("/images/"):
            path = os.path.join(IMG_DIR, os.path.basename(img_url))
            data = open(path, "rb").read()
        elif img_url.startswith("http"):
            data = urllib.request.urlopen(img_url, timeout=5).read()
        else:
            # use first post image
            db = get_db()
            row = db.execute("SELECT image_url FROM posts WHERE image_url IS NOT NULL LIMIT 1").fetchone()
            db.close()
            if not row: return {"error": "no posts with images"}
            path = os.path.join(IMG_DIR, os.path.basename(row["image_url"]))
            data = open(path, "rb").read()
            img_url = row["image_url"]

        from starlette.datastructures import UploadFile as StarletteUpload
        import io as _io

        # call the logic directly
        frame_img  = Image.open(_io.BytesIO(data)).convert("RGB")
        W, H       = frame_img.size
        proc, model = _load_florence()
        import torch, re

        def _run(task, image, text_input=None, max_new=300):
            prompt = task if text_input is None else task + text_input
            inputs = proc(text=prompt, images=image, return_tensors="pt")
            with torch.no_grad():
                ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new, num_beams=3, do_sample=False,
                    early_stopping=False,
                )
            raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
            return proc.post_process_generation(raw, task=task, image_size=(W, H))

        caption = _run("<MORE_DETAILED_CAPTION>", frame_img).get("<MORE_DETAILED_CAPTION>","")
        od_raw  = _run("<OD>", frame_img)
        od_data = od_raw.get("<OD>", {})
        detections = list(zip(od_data.get("labels",[]), od_data.get("bboxes",[])))

        return {
            "img_url":    img_url,
            "target":     target,
            "caption":    caption,
            "detections": [{"label": l, "bbox": b} for l,b in detections],
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
