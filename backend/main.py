
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import sqlite3, pickle, os, hmac, hashlib, json, uuid
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "recoverai.db"
MODEL_FILE = ROOT / "ml" / "recovery_model.pkl"
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(title="RecoverAI", version="1.0.0",
              description="AI Revenue Recovery demo backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
if MODEL_FILE.exists():
    with MODEL_FILE.open("rb") as f:
        MODEL = pickle.load(f)

def db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    con.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        customer TEXT,
        amount INTEGER,
        payment_method TEXT,
        failure_reason TEXT,
        previous_successes INTEGER DEFAULT 0,
        retry_count INTEGER DEFAULT 0,
        hour INTEGER DEFAULT 12,
        recovery_probability REAL DEFAULT 0.5,
        recommended_action TEXT,
        recommended_retry_time TEXT,
        status TEXT DEFAULT 'FAILED',
        created_at TEXT
    )
    """)
    con.commit()
    con.close()

def predict(payload):
    amount = float(payload.get("amount", 0))
    method = payload.get("payment_method", "UPI")
    reason = payload.get("failure_reason", "Bank timeout")
    prev = int(payload.get("previous_successes", 8))
    retries = int(payload.get("retry_count", 0))
    hour = int(payload.get("hour", datetime.now().hour))

    if MODEL is not None:
        import pandas as pd
        X = pd.DataFrame([{
            "amount": amount,
            "payment_method": method,
            "failure_reason": reason,
            "previous_successes": prev,
            "retry_count": retries,
            "hour": hour
        }])
        probability = float(MODEL.predict_proba(X)[0][1])
    else:
        probability = 0.5
        if reason in ["Bank timeout", "Network error"]: probability += .25
        if prev >= 10: probability += .15
        elif prev >= 5: probability += .10
        if retries == 0: probability += .05
        elif retries >= 2: probability -= .15
        probability = max(.05, min(.98, probability))

    if probability >= .80:
        action = "AUTO_RETRY"
        retry_time = "15 minutes"
    elif probability >= .50:
        action = "ALTERNATE_PAYMENT"
        retry_time = "30 minutes"
    else:
        action = "NOTIFY_CUSTOMER"
        retry_time = "Do not retry automatically"

    recommended_method = "UPI" if method != "UPI" and reason in ["Card declined", "Authentication failed"] else method
    risk = "LOW" if probability >= .80 else "MEDIUM" if probability >= .50 else "HIGH"

    return {
        "recovery_probability": round(probability, 4),
        "risk": risk,
        "recommended_action": action,
        "recommended_retry_time": retry_time,
        "recommended_method": recommended_method
    }

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def home():
    return {"name":"RecoverAI","status":"running","app":"/app","docs":"/docs"}

@app.get("/app")
def app_page():
    return FileResponse(FRONTEND)

@app.get("/api/health")
def health():
    return {"status":"healthy","service":"RecoverAI","model_loaded":MODEL is not None}

@app.get("/api/transactions")
def transactions():
    con = db()
    rows = con.execute("SELECT * FROM transactions ORDER BY created_at DESC").fetchall()
    con.close()
    if not rows:
        return {"count":0,"transactions":[]}
    return {"count":len(rows),"transactions":[dict(r) for r in rows]}

@app.post("/api/recovery/predict")
def recovery_predict(payload: dict):
    return predict(payload)

@app.post("/api/demo/payment")
def demo_payment(payload: dict):
    amount = int(payload.get("amount", 4999))
    customer = payload.get("customer", "Demo Customer")
    reasons = ["Bank timeout","Network error","Card declined"]
    reason = reasons[uuid.uuid4().int % len(reasons)]
    method = "UPI" if reason != "Card declined" else "Card"
    previous = 8 + (uuid.uuid4().int % 8)
    result = predict({
        "amount": amount,
        "payment_method": method,
        "failure_reason": reason,
        "previous_successes": previous,
        "retry_count": 0
    })
    tid = "TXN-" + str(90000 + (uuid.uuid4().int % 9999))
    now = datetime.utcnow().isoformat()
    con = db()
    con.execute("""INSERT INTO transactions
    (id,customer,amount,payment_method,failure_reason,previous_successes,retry_count,hour,
     recovery_probability,recommended_action,recommended_retry_time,status,created_at)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (tid,customer,amount,method,reason,previous,0,datetime.now().hour,
     result["recovery_probability"],result["recommended_action"],
     result["recommended_retry_time"],"FAILED",now))
    con.commit(); con.close()
    return {"transaction": {
        "id":tid,"customer":customer,"amount":amount,"failure_reason":reason,
        "recovery_probability":result["recovery_probability"],
        "recommended_action":result["recommended_action"],
        "recommended_retry_time":result["recommended_retry_time"],
        "recommended_method":result["recommended_method"]
    }}

@app.post("/api/recovery/recover")
def recover_payment(payload: dict):
    tid = payload.get("transaction_id","UNKNOWN")
    con = db()
    con.execute("UPDATE transactions SET status='RECOVERY_SCHEDULED' WHERE id=?", (tid,))
    con.commit(); con.close()
    return {"success":True,"transaction_id":tid,"status":"RECOVERY_SCHEDULED"}

@app.post("/api/webhook/razorpay")
async def razorpay_webhook(request: Request, x_razorpay_signature: str | None = Header(default=None)):
    body = await request.body()
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")
    if secret and x_razorpay_signature:
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, x_razorpay_signature):
            raise HTTPException(status_code=400, detail="Invalid webhook signature")

    payload = json.loads(body.decode("utf-8") or "{}")
    event = payload.get("event","unknown")
    return {"received":True,"event":event}

