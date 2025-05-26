import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# CSV 파일 불러오기 (pandas 안 씀)
def load_csv(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        for row in reader:
            features = list(map(float, row[:-1]))  # 마지막 열 빼고 전부 float 처리
            label = int(row[-1])                  # 마지막 열이 타겟값 (0 또는 1)
            X.append(features)
            y.append(label)
    return X, y

# CSV 파일에서 데이터 로드
X, y = load_csv('smoking_data.csv')

# 학습용/테스트용 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 평가 결과 출력
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
