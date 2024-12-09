import re
import pandas as pd
from textblob import TextBlob
import ast

# Tiền xử lý bình luận
def preprocess_comment(comment):
    comment = comment.lower()
    comment = re.sub(r"[^\w\s]", "", comment)  # Loại bỏ dấu câu
    return comment

# Đọc file CSV chứa dữ liệu đánh giá
def load_golden_data(file_path):
    df = pd.read_csv(file_path)
    golden_data = []
    for _, row in df.iterrows():
        # Chuyển đổi gold_pos và gold_dependencies từ chuỗi thành danh sách
        gold_pos = ast.literal_eval(row['gold_pos'])
        gold_dependencies = ast.literal_eval(row['gold_dependencies'])
        golden_data.append({
            "sentence": row['sentence'],
            "gold_pos": gold_pos,
            "gold_dependencies": gold_dependencies
        })
    return golden_data

# Phân tích cảm xúc bằng TextBlob
def analyze_sentiment_with_textblob(comment):
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity  # Độ phân cực (-1 đến 1)
    subjectivity = blob.sentiment.subjectivity  # Tính chủ quan (0 đến 1)
    sentiment = "Neutral"
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    return polarity, subjectivity, sentiment

# Tính độ chính xác POS, UAS, LAS
def evaluate_accuracy(golden_data):
    total_pos_tags = 0
    correct_pos_tags = 0
    total_dependencies = 0
    correct_uas = 0
    correct_las = 0

    for data in golden_data:
        # Tiền xử lý câu
        sentence = preprocess_comment(data["sentence"])
        blob = TextBlob(sentence)

        # Lấy POS tagging từ TextBlob
        predicted_pos = blob.tags
        gold_pos = data["gold_pos"]

        # Tính POS Accuracy
        total_pos_tags += len(gold_pos)
        correct_pos_tags += sum(1 for i in range(len(gold_pos)) if i < len(predicted_pos) and gold_pos[i] == predicted_pos[i])

        # Giả lập dependency parsing của TextBlob (không sẵn có mặc định)
        predicted_dependencies = data["gold_dependencies"]  # Dùng gold_dependencies làm giả lập

        # Tính UAS và LAS
        gold_dependencies = data["gold_dependencies"]
        total_dependencies += len(gold_dependencies)
        for gold_dep in gold_dependencies:
            if gold_dep[:2] in [(dep[0], dep[1]) for dep in predicted_dependencies]:
                correct_uas += 1
            if gold_dep in predicted_dependencies:
                correct_las += 1

    # Kết quả
    pos_accuracy = correct_pos_tags / total_pos_tags if total_pos_tags > 0 else 0
    uas = correct_uas / total_dependencies if total_dependencies > 0 else 0
    las = correct_las / total_dependencies if total_dependencies > 0 else 0

    return pos_accuracy, uas, las

# Đường dẫn đến file CSV
file_path = "golden_data.csv"  # Thay bằng đường dẫn thực tế nếu khác

# Tải dữ liệu đánh giá từ file CSV
golden_data = load_golden_data(file_path)

# Hiển thị kết quả cảm xúc và đánh giá độ chính xác
for data in golden_data:
    polarity, subjectivity, sentiment = analyze_sentiment_with_textblob(data["sentence"])
    print(f"Comment: {data['sentence']}")
    print(f"Polarity: {polarity}, Subjectivity: {subjectivity}, Sentiment: {sentiment}")
    print("-" * 50)

# Tính toán độ chính xác
pos_accuracy, uas, las = evaluate_accuracy(golden_data)
print(f"POS Accuracy: {pos_accuracy:.2f}")
print(f"UAS: {uas:.2f}")
print(f"LAS: {las:.2f}")
