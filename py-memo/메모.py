# ======================================== 
# 웨이퍼 페이지 시각화/UI 기능 정립
# ======================================== 



# ======================================== 
# 웨이퍼 관련 컬럼명 정하기
# ======================================== 
wafer_data = [
    lotName,          # wafer lot 이름 (식별자)
    waferMap,         # 분석 완료된 wafer를 Base64 형태로 저장
    failureType,      # AI 예측 불량 유형 (Label)
    confidence,       # AI 확신도 (Score)
    secFailureType,   # AI 예측 불량 유형 (Label)
    dieCount,         # 전체 칩 개수
    defectCount,      # 불량 칩 개수
    defectDensity,    # 불량 밀도 (불량 칩 개수/전체 칩 개수)
    totalGrade,       # 전체 등급 (A, B, C, D, F)
    created_at,       # 생성일 (DB Insert Time)
]