FDS V6 교육용 Model Serving MVP 실행 순서
=========================================

1. 아래 파일을 같은 프로젝트 루트에 둡니다.
   - 01_03_FDS_V6_통합_모델개발_DB_EDA_성능향상_1줄정리_서비스연결수정.ipynb
   - 04_FDS_V6_ModelServing_Streamlit_Docker.ipynb
   - app_realtime_fds.py
   - Dockerfile.streamlit
   - docker-compose.local.yml
   - requirements.streamlit.txt

2. 03 모델개발 노트북을 마지막 셀까지 실행합니다.
   생성 파일:
   models/fds_model.joblib
   models/fds_model_metadata.json
   models/fds_hybrid_metadata.json
   models/fds_rule_config.json
   models/fds_model.sha256
   output/fds_realtime_demo.csv.gz
   requirements.streamlit.txt  (학습 환경 버전으로 자동 갱신)

3. 04 Serving 노트북을 실행해 Artifact SHA256, Runtime 버전, 1건/Batch 추론을 확인합니다.

4. 로컬 Streamlit 실행
   pip install -r requirements.streamlit.txt
   streamlit run app_realtime_fds.py
   http://localhost:8501

5. Docker 실행
   docker compose -f docker-compose.local.yml build --no-cache
   docker compose -f docker-compose.local.yml up -d
   docker compose -f docker-compose.local.yml ps
   docker compose -f docker-compose.local.yml logs -f fds-mvp
   http://localhost:8501

주의
----
- 현재 MVP는 03에서 Feature Engineering이 끝난 Test 대표 거래를 서비스처럼 재생하는 교육용 구조입니다.
- 실제 운영에서는 신규 원시 거래 -> Online Feature Engineering/Feature Store -> Rule Score -> ML -> Hybrid 단계가 추가로 필요합니다.
- fraud_label/fraud_type은 교육용 정답 확인에만 사용하며 모델 입력/의사결정에는 사용하지 않습니다.
- 03이 생성한 requirements.streamlit.txt를 사용하면 모델 학습 환경과 Docker/Streamlit의 핵심 라이브러리 버전을 맞출 수 있습니다.
