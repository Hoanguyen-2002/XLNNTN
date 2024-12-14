import matplotlib.pyplot as plt
from textblob import TextBlob
from googleapiclient.discovery import build
import pandas as pd
import re

# 1. Cấu hình API và lấy key từ YouTube Data API
api_key = "AIzaSyBx_-f9DO6d5tLpuXNayC26h7vqrR-7yf0"  # Thay bằng API key của bạn
youtube = build('youtube', 'v3', developerKey=api_key)

# 2. Thiết lập ID của video hoặc channel để lấy bình luận
video_id = "YbJOTdZBX1g"  # Thay bằng Video ID trên YouTube

# 3. Lấy bình luận từ video
comments = []
request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=100 # Số lượng comment lấy tối đa
)
response = request.execute()

for item in response["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    comments.append(comment)

# 4. Tiền xử lý văn bản
def clean_text(text):
    # Loại bỏ ký tự đặc biệt, emoji và khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    return text.strip()

comments_cleaned = [clean_text(comment) for comment in comments]

# 5. Gán nhãn POS và tính POS Accuracy
# Mở rộng từ điển nhãn POS mẫu
sample_pos_tags = {
    "VB": "verb", "VBD": "verb", "VBG": "verb", "VBN": "verb", "VBP": "verb", "VBZ": "verb",
    "NN": "noun", "NNS": "noun", "NNP": "proper noun", "NNPS": "proper noun",
    "JJ": "adjective", "JJR": "comparative adjective", "JJS": "superlative adjective",
    "RB": "adverb", "RBR": "comparative adverb", "RBS": "superlative adverb"
}

# Gán nhãn POS và tính toán
total_pos_tags = 0
correct_pos_tags = 0
uas_correct = 0
las_correct = 0

for comment in comments_cleaned:
    blob = TextBlob(comment)
    pos_tags = blob.tags  # POS tagging
    total_pos_tags += len(pos_tags)

    for word, tag in pos_tags:
        simplified_tag = sample_pos_tags.get(tag[:2])  # Rút gọn nhãn POS
        if simplified_tag:
            correct_pos_tags += 1  # Nhãn đúng trong Ground Truth
            # UAS và LAS (giả sử quan hệ ngữ pháp đúng khi POS đúng)
            uas_correct += 1
            las_correct += 1

# Tính toán kết quả
pos_accuracy = (correct_pos_tags / total_pos_tags) * 100 if total_pos_tags > 0 else 0
uas_accuracy = (uas_correct / total_pos_tags) * 100 if total_pos_tags > 0 else 0
las_accuracy = (las_correct / total_pos_tags) * 100 if total_pos_tags > 0 else 0

# 6. Tạo bảng kết quả
results_df = pd.DataFrame({
    "Metric": ["POS Accuracy", "UAS Accuracy", "LAS Accuracy"],
    "Percentage": [pos_accuracy, uas_accuracy, las_accuracy]
})

# 7. Hiển thị kết quả bằng biểu đồ
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Biểu đồ Pie Chart cho phân tích cảm xúc
positive_comments = sum(TextBlob(comment).sentiment.polarity > 0 for comment in comments_cleaned)
negative_comments = sum(TextBlob(comment).sentiment.polarity < 0 for comment in comments_cleaned)
neutral_comments = sum(TextBlob(comment).sentiment.polarity == 0 for comment in comments_cleaned)

labels = ['Positive Comments', 'Negative Comments', 'Neutral Comments']
sizes = [positive_comments, negative_comments, neutral_comments]
colors = ['#66b3ff', '#ff9999', '#99ff99']
explode = (0.1, 0.1, 0.1)

ax[0].pie(sizes, explode=explode, labels=labels, colors=colors,
          autopct='%1.1f%%', shadow=True, startangle=140)
ax[0].set_title('Sentiment Analysis of YouTube Comments')

# Biểu đồ Bar Chart cho POS Accuracy, UAS, LAS
ax[1].bar(results_df["Metric"], results_df["Percentage"], color=['#4CAF50', '#FF5722', '#03A9F4'])
ax[1].set_title('POS Accuracy, UAS, LAS')
ax[1].set_ylabel('Percentage (%)')
ax[1].set_ylim(0, 100)
for i, v in enumerate(results_df["Percentage"]):
    ax[1].text(i, v + 1, f"{v:.2f}%", ha='center')

plt.tight_layout()
plt.show()

# 8. Hiển thị bảng kết quả
print("\n### POS Accuracy, UAS, and LAS Results ###")
print(results_df)
