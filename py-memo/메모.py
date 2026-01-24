# ======================================== 
# 웨이퍼 관련 컬럼명 정하기
# ======================================== 
wafer_data = [
    lotName,          # wafer lot 이름 (식별자)
    waferMap,         # 분석 완료된 wafer를 Base64 형태로 저장
    waferSize,        # wafer 원본 이미지 크기 (AxB 느낌의 형태)
    failureType,      # [핵심] AI 예측 불량 유형 (Label)
    confidence,       # AI 확신도 (Score)
    dieCount,         # 전체 칩 개수
    defectCount,      # 불량 칩 개수
    defectDensity,    # 불량 밀도 (불량 칩 개수/전체 칩 개수)
    totalGrade,       # 전체 등급 (A, B, C, D, F)
    created_at        # 생성일 (DB Insert Time)
]

# ======================================== 
# 칩 관련 컬럼명 정하기
# ======================================== 
chip_data = [
    chip_uid,       # f"{lot_name}X{x}Y{y}D{die_status}" 형태
    lotName,        # [핵심] 부모 웨이퍼의 lotName
    tsv_matrix,     # np.zeros(chip_grid_size, dtype=int) 형태
    tsv_coordinate, # 부모 웨이퍼로부터 추출될 당시 이 칩의 (x,y) 좌표
    tsv_status,     # 0 (정상) 또는 1 (불량), int(die_status)
    created_at      # 생성일 (DB Insert Time)
]