from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer, util
from pyvi.ViTokenizer import tokenize

# ================== KHỞI TẠO MODEL ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("dangvantuan/vietnamese-embedding").to(device)

# Dataset ngành học
majors = {
    "Nấu ăn (Đầu bếp)": "Chế biến món ăn, nghệ thuật ẩm thực, trang trí đồ ăn",
    "CNTT": "Lập trình, máy tính, phát triển phần mềm, trí tuệ nhân tạo",
    "Marketing": "Truyền thông, quảng cáo, sáng tạo nội dung, thị trường",
    "Thiết kế đồ họa": "Mỹ thuật, sáng tạo, thiết kế hình ảnh, màu sắc",
    "Du lịch": "Khám phá, địa điểm du lịch, phục vụ khách hàng",
}

# Encode trước mô tả ngành để tăng tốc
major_sentences = [tokenize(desc) for desc in majors.values()]
emb_majors = model.encode(major_sentences, convert_to_tensor=True, device=device)

# ================== API ==================
app = FastAPI()


class UserInput(BaseModel):
    text: str


@app.post("/predict")
def predict_major(data: UserInput):
    user_tok = tokenize(data.text)
    emb_user = model.encode(user_tok, convert_to_tensor=True, device=device)

    # Tính điểm giống nhau
    scores = util.cos_sim(emb_user, emb_majors)[0]

    # Tạo danh sách kết quả trả về
    results = []
    for (name, _), score in zip(majors.items(), scores):
        results.append({
            "major": name,
            "score": float(score)
        })

    best_major = list(majors.keys())[torch.argmax(scores)]

    return {
        "best_major": best_major,
        "scores": results
    }
