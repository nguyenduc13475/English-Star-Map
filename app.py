import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone

import aiofiles
import groq
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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

STATS_FILE = "hierarchy_stats.json"
if os.path.exists(STATS_FILE):
    with open(STATS_FILE, "r") as f:
        hierarchy_stats = json.load(f)
else:
    hierarchy_stats = {"l1": {}, "l2": {}, "l3": {}}
    for n in nodes:
        p = progress_data[str(n["id"])]
        m = p["mastery"]
        for level, cluster in [
            ("l1", n.get("l1_cluster")),
            ("l2", n.get("l2_cluster")),
            ("l3", n.get("l3_cluster")),
        ]:
            if cluster == -1 or cluster is None:
                continue
            c_str = str(cluster)
            if c_str not in hierarchy_stats[level]:
                hierarchy_stats[level][c_str] = {
                    "mastery": 0,
                    "maxMastery": -1,
                    "topWord": "",
                }
            hierarchy_stats[level][c_str]["mastery"] += m
            if m >= hierarchy_stats[level][c_str]["maxMastery"]:
                hierarchy_stats[level][c_str]["maxMastery"] = m
                hierarchy_stats[level][c_str]["topWord"] = n["chunk"]
    with open(STATS_FILE, "w") as f:
        json.dump(hierarchy_stats, f)

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
        return (
            -1
        )  # Chưa học bao giờ, đưa ra khỏi danh sách ưu tiên của Space Repetition
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
    phase = data.get("phase", 1)  # Mặc định là Phase 1 & 2

    if phase == 0:
        system_prompt = """
Mày là một ông thầy dạy tiếng Anh vô cùng nghiêm khắc, giang hồ nhưng nói chuyện RẤT TỰ NHIÊN, đời thường.
NHIỆM VỤ CỦA MÀY:
Tao đang gặp cụm từ này lần ĐẦU TIÊN. Hãy giải thích ý nghĩa nôm na, dễ hiểu của nó bằng tiếng Việt, sau đó đưa ra 1 ví dụ tiếng Anh chứa cụm từ đó cực kỳ bựa để minh họa, đồng thời giải thích nghĩa của ví dụ đó luôn (ĐẶC BIỆT phải giải thích TENSE được dùng trong ví dụ sao cho hợp lý).
TUYỆT ĐỐI CHỈ GIẢI THÍCH, KHÔNG ĐƯỢC ĐẶT CÂU HỎI. KHÔNG BẮT TAO TRẢ LỜI. MÀY NÓI XONG LÀ HẾT NHIỆM VỤ.
*LƯU Ý ĐẶC BIỆT*: BẮT BUỘC bọc TẤT CẢ mọi cụm từ rời rạc tiếng Anh, và câu tiếng Anh bằng thẻ XML <en> và </en>. KHÔNG ĐƯỢC ĐỂ SÓT CHỮ TIẾNG ANH NÀO Ở NGOÀI THẺ. TUYỆT ĐỐI KHÔNG dùng thẻ này cho tiếng Việt.

QUY TẮC PHẢN HỒI JSON:
Trả về duy nhất 1 chuỗi JSON hợp lệ với 2 field:
- "response": Lời mày dạy tao.
- "score": -2
"""
    else:
        system_prompt = """
Mày là một ông thầy dạy tiếng Anh vô cùng nghiêm khắc, giang hồ nhưng nói chuyện RẤT TỰ NHIÊN, đời thường, ngắn gọn. 
QUY TRÌNH HỌC (2 Phase liên tiếp):
- Phase 1 (Đoán nghĩa): Mày sinh ra 1 câu tiếng Anh ngẫu nhiên chứa cụm từ target. Hỏi tao nghĩa cụm từ đó. Chờ tao trả lời. Nếu sai/lạc đề, chửi sấp mặt và giải thích ngắn gọn. Nếu đúng, khen mỉa mai.
- Phase 2 (Tạo câu): Mày tạo 1 tình huống thực tế bựa bắt tao dùng cụm target để đối đáp. Nếu tao dùng sai, chửi thậm tệ và đưa ra lời khuyên sâu sắc.
*LƯU Ý ĐẶC BIỆT 1*: Nếu tao gửi tin nhắn "[SYSTEM_SILENCE]", nghĩa là tao đang câm. Hãy chửi tao chậm chạp và giục tao.
*LƯU Ý ĐẶC BIỆT 2*: BẮT BUỘC bọc TẤT CẢ cụm từ rời rạc tiếng Anh, và mọi câu tiếng Anh bằng thẻ XML <en> và </en>. KHÔNG ĐƯỢC ĐỂ SÓT CHỮ TIẾNG ANH NÀO Ở NGOÀI THẺ.

QUY TẮC PHẢN HỒI JSON:
Trả về duy nhất 1 chuỗi JSON hợp lệ với 2 field:
- "response": Lời mày nói với tao.
- "score": Điểm số (-1 khi chưa chốt, 0-10 khi hoàn thành Phase 2).
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

    # Propagate stats theo cấp bậc
    n_updated = nodes[int(node_id)]
    for level, cluster in [
        ("l1", n_updated.get("l1_cluster")),
        ("l2", n_updated.get("l2_cluster")),
        ("l3", n_updated.get("l3_cluster")),
    ]:
        if cluster == -1 or cluster is None:
            continue
        c_str = str(cluster)
        cluster_nodes = [x for x in nodes if x.get(f"{level}_cluster") == cluster]
        tot, m_max, top_w = 0, -1, ""
        for cn in cluster_nodes:
            c_prog = progress_data[str(cn["id"])]["mastery"]
            tot += c_prog
            if c_prog >= m_max:
                m_max = c_prog
                top_w = cn["chunk"]
        hierarchy_stats[level][c_str] = {
            "mastery": tot,
            "maxMastery": m_max,
            "topWord": top_w,
        }
    async with aiofiles.open(STATS_FILE, "w", encoding="utf-8") as f:
        await f.write(json.dumps(hierarchy_stats, ensure_ascii=False))

    # Dọn dẹp toàn bộ file audio tạm trong static/audio sau mỗi session hoàn thành
    audio_dir = "static/audio"
    if os.path.exists(audio_dir):
        for file in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:  # noqa: BLE001
                print(f"[Cleanup Warning] Không thể xóa file {file_path}: {e}")

    return {"status": "ok"}


@app.get("/api/stats")
def get_stats(level: str):
    return hierarchy_stats.get(level, {})


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

    # Chống ngợp trình duyệt: Giới hạn 15k điểm, ưu tiên tuyệt đối giữ lại các từ đã học (mastery > 0)
    if level != "l3" and len(filtered_idx) > 15000:
        learned_idx = [
            i for i in filtered_idx if progress_data[str(nodes[i]["id"])]["mastery"] > 0
        ]
        unlearned_idx = [
            i
            for i in filtered_idx
            if progress_data[str(nodes[i]["id"])]["mastery"] == 0
        ]
        max_unlearned = max(0, 15000 - len(learned_idx))
        if len(unlearned_idx) > max_unlearned:
            unlearned_idx = random.sample(unlearned_idx, max_unlearned)
        filtered_idx = learned_idx + unlearned_idx

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
                "days_passed": -1
                if p["last_studied"] == 0
                else (time.time() - p["last_studied"]) / 86400,
                "study_count": p["study_count"],
                "block": p["block_threshold"],
            }
        )

    total_mastery = get_total_mastery()
    return {"points": points, "total_mastery": total_mastery}


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
