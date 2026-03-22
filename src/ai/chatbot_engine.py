import pandas as pd

def get_chatbot_response(user_query: str, df: pd.DataFrame) -> str:
    """
    Hàm xử lý câu hỏi của người dùng dựa trên dữ liệu lịch sử trạm Phú Bài.
    Sử dụng Rule-based (Từ khóa) cho giai đoạn 1.
    """
    if not isinstance(user_query, str) or df.empty:
        return "Xin lỗi, hệ thống đang gặp lỗi dữ liệu hoặc câu hỏi không hợp lệ."

    # Chuẩn hóa câu hỏi: viết thường, xóa khoảng trắng thừa
    q = user_query.lower().strip()

    # 1. Nhóm từ khóa: Chào hỏi
    if q in ["chào", "hello", "hi", "chào bot", "chào bạn"]:
        return "👋 Chào bạn! Tôi là trợ lý AI chuyên phân tích dữ liệu trạm Phú Bài. Bạn muốn tra cứu kỷ lục nhiệt độ hay lượng mưa?"

    # 2. Nhóm từ khóa: Nhiệt độ cao nhất (Nóng nhất)
    if any(kw in q for kw in ["nóng nhất", "nhiệt độ cao nhất", "kỷ lục nóng", "nhiệt độ max"]):
        max_row = df.loc[df['max_temp_c'].idxmax()]
        val = max_row['max_temp_c']
        day = max_row['date'].strftime('%d/%m/%Y')
        return f"🔥 Ngày nóng nhất tại Phú Bài ghi nhận được là **{val}°C** vào ngày {day}."

    # 3. Nhóm từ khóa: Nhiệt độ thấp nhất (Lạnh nhất)
    if any(kw in q for kw in ["lạnh nhất", "rét nhất", "nhiệt độ thấp nhất", "kỷ lục lạnh"]):
        min_row = df.loc[df['min_temp_c'].idxmin()]
        val = min_row['min_temp_c']
        day = min_row['date'].strftime('%d/%m/%Y')
        return f"❄️ Ngày lạnh nhất tại Phú Bài ghi nhận được là **{val}°C** vào ngày {day}."

    # 4. Nhóm từ khóa: Lượng mưa lớn nhất (Kỷ lục mưa)
    if any(kw in q for kw in ["mưa to nhất", "mưa nhiều nhất", "kỷ lục mưa", "lượng mưa lớn nhất"]):
        max_rain_row = df.loc[df['precipitation_mm'].idxmax()]
        val = max_rain_row['precipitation_mm']
        day = max_rain_row['date'].strftime('%d/%m/%Y')
        return f"⛈️ Lượng mưa kỷ lục tại trạm Phú Bài lên tới **{val} mm** vào ngày {day}. Hôm đó trời đổ mưa rất lớn!"

    # 5. Fallback: Khi bot không hiểu câu hỏi
    return (
        "🤖 Xin lỗi, hiện tại bộ não của tôi chỉ mới được huấn luyện để tra cứu các kỷ lục:\n"
        "- **Nhiệt độ cao nhất**\n"
        "- **Nhiệt độ thấp nhất**\n"
        "- **Lượng mưa kỷ lục**\n\n"
        "Bạn hãy thử hỏi bằng các từ khóa trên nhé! (Sắp tới tôi sẽ được nâng cấp bằng Gemini AI 🚀)"
    )


# =====================================================================
# PHẦN CHUẨN BỊ CHO GIAI ĐOẠN 2: TÍCH HỢP GEMINI API LÊN ĐÂY
# =====================================================================
def get_gemini_response(user_query: str, df_context: pd.DataFrame, api_key: str) -> str:
    """
    (Hàm dự phòng) Gửi dữ liệu trạm Phú Bài lên Google Gemini để AI tự phân tích và trả lời.
    Bạn sẽ code phần này khi lấy được API Key.
    """
    pass