import hashlib
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone

import edge_tts
import groq
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import Groq

load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
os.makedirs("static", exist_ok=True)
os.makedirs("static/audio", exist_ok=True)
os.makedirs("templates", exist_ok=True)
with open("galaxy_nodes.json", "r", encoding="utf-8") as f:
    nodes = json.load(f)
embeddings_2d = np.load("embeddings_2d.npy")

PROGRESS_FILE = "user_progress.json"
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress_data = json.load(f)
else:
    progress_data = {
        str(n["id"]): {
            "mastery": 0,
            "study_count": 0,
            "last_studied": 0,
            "block_threshold": 0,
        }
        for n in nodes
    }

session_state = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "count_today": 0,
}


def save_progress():
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f)


def get_total_mastery():
    return sum(d["mastery"] for d in progress_data.values())


def calc_forgetting(node_id):
    p = progress_data[str(node_id)]
    if p["last_studied"] == 0:
        return 9999  # Chưa học bao giờ
    days_passed = (time.time() - p["last_studied"]) / 86400
    return days_passed / (p["mastery"] + 1)


@app.get("/api/next_phrase")
def get_next_phrase():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if session_state["date"] != today:
        session_state["date"] = today
        session_state["count_today"] = 0

    current_progress = get_total_mastery()
    available_nodes = [
        n
        for n in nodes
        if progress_data[str(n["id"])]["block_threshold"] <= current_progress
    ]

    if not available_nodes:
        return {"error": "Tất cả các từ đã bị Block hoặc chưa có data!"}

    chosen_node = None

    if session_state["count_today"] < 10:
        # Chiến thuật Breadth-First: Phủ 10 Đại tinh
        target_l1 = session_state["count_today"] % 10
        l1_nodes = [n for n in available_nodes if n.get("l1_cluster") == target_l1]
        if not l1_nodes:
            l1_nodes = available_nodes  # Fallback

        # Tìm L2 có tổng mastery thấp nhất, tie-breaker: khoảng cách tới trung tâm L1
        l2_stats = defaultdict(lambda: {"mastery": 0, "dist": 0.0})
        for n in l1_nodes:
            l2 = n.get("l2_cluster", -1)
            l2_stats[l2]["mastery"] += progress_data[str(n["id"])]["mastery"]
            l2_stats[l2]["dist"] += n.get("dist_to_l1", 999.0)
        min_l2 = min(
            l2_stats.keys(), key=lambda k: (l2_stats[k]["mastery"], l2_stats[k]["dist"])
        )
        l2_nodes = [n for n in l1_nodes if n.get("l2_cluster") == min_l2]

        # Tìm L3 có tổng mastery thấp nhất, tie-breaker: khoảng cách tới trung tâm L2
        l3_stats = defaultdict(lambda: {"mastery": 0, "dist": 0.0})
        for n in l2_nodes:
            l3 = n.get("l3_cluster", -1)
            l3_stats[l3]["mastery"] += progress_data[str(n["id"])]["mastery"]
            l3_stats[l3]["dist"] += n.get("dist_to_l2", 999.0)
        min_l3 = min(
            l3_stats.keys(), key=lambda k: (l3_stats[k]["mastery"], l3_stats[k]["dist"])
        )
        l3_nodes = [n for n in l2_nodes if n.get("l3_cluster") == min_l3]

        # Chọn node có mastery thấp nhất, tie-breaker: dist_to_l3
        chosen_node = min(
            l3_nodes,
            key=lambda x: (
                progress_data[str(x["id"])]["mastery"],
                x.get("dist_to_l3", 999.0),
            ),
        )
    else:
        # Cơ chế Space Repetition (Forgetting Curve)
        chosen_node = max(available_nodes, key=lambda x: calc_forgetting(x["id"]))

    session_state["count_today"] += 1
    return {"node": chosen_node, "stats": progress_data[str(chosen_node["id"])]}


@app.post("/api/chat")
async def chat_ai(req: Request):
    data = await req.json()
    history = data.get("history", [])
    system_prompt = """
Mày là một giáo viên tiếng Anh cực kỳ mỏ hỗn, cục súc, văng tục thoải mái (fuck, shit, ngu đần,...) nhưng dạy kiến thức cực kỳ chuẩn xác.
QUY TRÌNH HỌC (2 Phase liên tiếp):
- Phase 1 (Đoán nghĩa): Mày sinh ra 1 câu tiếng Anh ngẫu nhiên chứa cụm từ target. Hỏi tao nghĩa cụm từ đó. Chờ tao trả lời. Nếu sai/lạc đề, chửi sấp mặt và giải thích đúng. Nếu đúng, khen kiểu bố đời. 
- Phase 2 (Tạo câu): Mày tạo 1 tình huống thực tế bựa/đời thường bắt tao dùng cụm target để đối đáp. Nếu tao dùng sai thì (tense) hoặc sai ngữ pháp, chửi thậm tệ.
*LƯU Ý ĐẶC BIỆT*: Nếu tao gửi tin nhắn có nội dung "[SYSTEM_SILENCE]", có nghĩa là tao đang câm như hến không trả lời kịp. Hãy chửi tao là chậm chạp, ngu đần, lề mề và giục tao trả lời nhanh lên.

QUY TẮC PHẢN HỒI JSON:
Trả về duy nhất 1 chuỗi JSON hợp lệ với 2 field:
- "response": Lời mày nói với tao (Văn phong mỏ hỗn, tự nhiên, sinh động để tao nghe qua Audio).
- "score": Điểm số. Nếu tao yêu cầu giải thích từ mới (Phase 0), trả về -2. Nếu đang ở Phase 1 hoặc chưa xong Phase 2, để -1. Nếu tao đã hoàn thành đối đáp Phase 2, chốt điểm (0-10).
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        reply = json.loads(completion.choices[0].message.content)
    except groq.GroqError as e:
        reply = {"response": f"Lỗi Groq API: {e!s}", "score": -1}
    return reply


@app.post("/api/update_node")
async def update_node(req: Request):
    data = await req.json()
    node_id = str(data["id"])
    score = data["score"]  # Điểm từ 0-10, hoặc -2 cho từ mới
    p = progress_data[node_id]
    p["last_studied"] = time.time()

    if score == -2:  # Lần đầu học
        p["mastery"] = 0.5  # Giá trị định sẵn thấp
        p["study_count"] = 1
    elif score >= 7:
        p["mastery"] += (score - 5) * 0.5
        p["study_count"] = 0
    else:
        p["mastery"] = max(0, p["mastery"] - 1)
        p["study_count"] += 1

    if p["study_count"] >= 5:  # Học ngu 5 lần
        current_prog = get_total_mastery()
        p["block_threshold"] = current_prog + 50  # Đợi đủ trình mới mở
        p["study_count"] = 0
        p["mastery"] = 0

    save_progress()
    return {"status": "ok"}


@app.get("/api/tts")
async def generate_tts(text: str):
    # Chọn giọng. Có thể đổi sang en-US-JennyNeural (Nữ) hoặc en-US-TonyNeural (Nam)
    voice = "en-US-GuyNeural"
    # Tạo tên file hash để tái sử dụng nếu câu nói trùng nhau, tránh gen lại
    text_hash = hashlib.md5(text.encode()).hexdigest()
    file_path = f"static/audio/{text_hash}.mp3"

    if not os.path.exists(file_path):
        communicate = edge_tts.Communicate(
            text, voice, rate="+10%"
        )  # rate tăng tốc độ một chút cho tự nhiên
        await communicate.save(file_path)

    return FileResponse(file_path, media_type="audio/mpeg")


@app.get("/api/map_data")
def get_map_data(level: str = "root", l1: int = -1, l2: int = -1, l3: int = -1):
    filtered_idx = []
    for i, n in enumerate(nodes):
        if (
            level == "root"
            or level == "l1"
            and n.get("l1_cluster") == l1
            or level == "l2"
            and n.get("l2_cluster") == l2
            or level == "l3"
            and n.get("l3_cluster") == l3
        ):
            filtered_idx.append(i)

    # Chống ngợp trình duyệt: Giới hạn 15k điểm cho tất cả các cấp, trừ cấp chòm sao (l3)
    if level != "l3" and len(filtered_idx) > 15000:
        filtered_idx = random.sample(filtered_idx, 15000)

    max_mastery = max((d["mastery"] for d in progress_data.values()), default=1) or 1

    points = []
    for i in filtered_idx:
        n = nodes[i]
        p = progress_data[str(n["id"])]
        points.append(
            {
                "id": n["id"],
                "chunk": n["chunk"],
                "x": float(embeddings_2d[i][0]),
                "y": float(embeddings_2d[i][1]),
                "l1": n.get("l1_cluster"),
                "l2": n.get("l2_cluster"),
                "l3": n.get("l3_cluster"),
                "mastery": p["mastery"],
                "norm_mastery": p["mastery"] / max_mastery,
                "forgetting": calc_forgetting(n["id"]),
                "study_count": p["study_count"],
                "block": p["block_threshold"],
            }
        )
    return points


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
