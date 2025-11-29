from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
from sentence_transformers import SentenceTransformer, util
from pyvi.ViTokenizer import tokenize
import os

app = FastAPI(title="Gợi ý ngành học theo sở thích")

# Load model khi khởi động (chỉ load 1 lần)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("dangvantuan/vietnamese-embedding")
model = model.to(device)

# Dataset ngành học
majors = {
    "Nấu ăn (Đầu bếp)": "Chế biến món ăn, nghệ thuật ẩm thực, trang trí đồ ăn",
    "CNTT": "Lập trình, máy tính, phát triển phần mềm, trí tuệ nhân tạo",
    "Marketing": "Truyền thông, quảng cáo, sáng tạo nội dung, thị trường",
    "Thiết kế đồ họa": "Mỹ thuật, sáng tạo, thiết kế hình ảnh, màu sắc",
    "Du lịch": "Khám phá, địa điểm du lịch, phục vụ khách hàng",
}

major_sentences = [tokenize(desc) for desc in majors.values()]
emb_majors = model.encode(major_sentences, convert_to_tensor=True, device=device)

@app.get("/")
async def home():
    return {"message": "API gợi ý ngành học tiếng Việt sẵn sàng!"}

@app.get("/recommend")
async def recommend(interest: str = "Sở thích ăn uống, thích nấu ăn và tìm hiểu món ngon"):
    user_tok = tokenize(interest)
    emb_user = model.encode(user_tok, convert_to_tensor=True, device=device)
    
    scores = util.cos_sim(emb_user, emb_majors)[0]
    
    results = []
    for (name, _), score in zip(majors.items(), scores):
        results.append({"major": name, "score": round(float(score), 4)})
    
    # Sắp xếp giảm dần
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    best = results[0]["major"]
    
    return {
        "user_interest": interest,
        "recommendation": best,
        "all_scores": results
    }