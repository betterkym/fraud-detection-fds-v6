from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


# =============================================================================
# 00. 기본 설정
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / 'output'
MODEL_DIR = PROJECT_DIR / 'models'

MODEL_PATH = MODEL_DIR / 'fds_model.joblib'
ML_META_PATH = MODEL_DIR / 'fds_model_metadata.json'
HYBRID_META_PATH = MODEL_DIR / 'fds_hybrid_metadata.json'
RULE_CONFIG_PATH = MODEL_DIR / 'fds_rule_config.json'
MODEL_HASH_PATH = MODEL_DIR / 'fds_model.sha256'
DATA_PATH = OUTPUT_DIR / 'fds_realtime_demo.csv.gz'

st.set_page_config(
    page_title='FDS V6 교육용 Hybrid MVP',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f9fc; }

    .block-container {
        max-width: 1480px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4e9f0;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 108px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e4e9f0;
        border-radius: 12px;
        overflow: hidden;
    }

    .fds-card {
        background: white;
        border: 1px solid #e4e9f0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .fds-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #162033;
        margin-bottom: 4px;
    }

    .fds-sub {
        color: #667085;
        font-size: 0.90rem;
    }

    .good {
        color: #067647;
        font-weight: 700;
    }

    .bad {
        color: #b42318;
        font-weight: 700;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 01. Artifact 로드
# =============================================================================
required_files = [MODEL_PATH, ML_META_PATH, HYBRID_META_PATH, RULE_CONFIG_PATH, MODEL_HASH_PATH, DATA_PATH]
missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error('필수 Serving Artifact가 없습니다.')
    st.code('\n'.join(missing_files))
    st.stop()

expected_hash = MODEL_HASH_PATH.read_text(encoding='utf-8').strip()
actual_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
if expected_hash != actual_hash:
    st.error('fds_model.joblib SHA256이 저장된 값과 다릅니다. 모델 파일이 변경되었거나 잘못 복사되었습니다.')
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_events():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['event_at'] = pd.to_datetime(df['event_at'], errors='coerce')
    return df


try:
    model = load_model()
    ml_meta = load_json(ML_META_PATH)
    hybrid_meta = load_json(HYBRID_META_PATH)
    rule_meta = load_json(RULE_CONFIG_PATH)
    events = load_events()
except Exception as exc:
    st.error('Serving Artifact 로드 중 오류가 발생했습니다.')
    st.exception(exc)
    st.stop()


MODEL_FEATURES = ml_meta['model_features']
NUMERIC_FEATURES = ml_meta.get('numeric_features', [])
CATEGORICAL_FEATURES = ml_meta.get('categorical_features', [])

ML_WEIGHT = float(hybrid_meta['ml_weight'])
RULE_WEIGHT = float(hybrid_meta['rule_weight'])
HYBRID_THRESHOLD = float(hybrid_meta['hybrid_threshold'])
FRAUD_ACTION_THRESHOLDS = hybrid_meta['fraud_action_thresholds']

current_versions = {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__}
for package, key in [('xgboost', 'xgboost'), ('lightgbm', 'lightgbm')]:
    try:
        current_versions[key] = version(package)
    except PackageNotFoundError:
        current_versions[key] = None
expected_versions = ml_meta.get('runtime_versions', {})
version_rows = []
for key in sorted(set(expected_versions) | set(current_versions)):
    training_v, serving_v = expected_versions.get(key), current_versions.get(key)
    ok = str(training_v).split('.')[:2] == str(serving_v).split('.')[:2] if key == 'python' else training_v == serving_v
    version_rows.append({'package': key, 'training': training_v, 'serving': serving_v, 'status': 'OK' if ok else 'CHECK'})
version_check = pd.DataFrame(version_rows)
version_mismatch = bool((version_check['status'] == 'CHECK').any()) if not version_check.empty else False

missing_features = [feature for feature in MODEL_FEATURES if feature not in events.columns]

if events.empty:
    st.error('Serving Demo Dataset이 비었습니다.')
    st.stop()

if missing_features:
    st.error('Demo Dataset에 모델 Feature가 없습니다.')
    st.code('\n'.join(missing_features))
    st.stop()


# =============================================================================
# 02. Serving 함수
# =============================================================================
def make_model_input(row):
    X = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in MODEL_FEATURES}])

    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype('float64')

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            s = X[col].astype('object')
            X[col] = s.where(pd.notna(s), np.nan)

    return X


def decide_action(score):
    if score >= FRAUD_ACTION_THRESHOLDS['block']:
        return 'BLOCK_AND_REVIEW'
    if score >= FRAUD_ACTION_THRESHOLDS['temporary_hold']:
        return 'TEMP_HOLD'
    if score >= FRAUD_ACTION_THRESHOLDS['step_up']:
        return 'STEP_UP_AUTH'
    return 'PASS'


def num(row, key, default=0.0):
    return float(pd.to_numeric(pd.Series([row.get(key, default)]), errors='coerce').fillna(default).iloc[0])


def predict_event(row):
    started = time.perf_counter()

    ml_risk_score = float(model.predict_proba(make_model_input(row))[:, 1][0])
    rule_score = num(row, 'rule_score', 0)
    rule_probability = float(np.clip(rule_score / 100, 0, 1))
    hybrid_score = ML_WEIGHT * ml_risk_score + RULE_WEIGHT * rule_probability

    return {
        'event_uid': str(row.get('event_uid', '')),
        'event_at': row.get('event_at'),
        'source_table': str(row.get('source_table', '')),
        'event_subtype': str(row.get('event_subtype', '')),
        'amount_abs': num(row, 'amount_abs', 0),
        'rule_score': rule_score,
        'ml_risk_score': ml_risk_score,
        'hybrid_score': float(hybrid_score),
        'alert': int(hybrid_score >= HYBRID_THRESHOLD),
        'action': decide_action(hybrid_score),
        'actual_label': int(num(row, 'fraud_label', 0)),
        'actual_fraud_type': str(row.get('fraud_type', 'NONE')),
        'is_new_device': int(num(row, 'is_new_device_event', 0)),
        'foreign_ip': int(num(row, 'foreign_ip_flag', 0)),
        'new_beneficiary': int(num(row, 'beneficiary_is_new', 0)),
        'beneficiary_risk': num(row, 'beneficiary_risk_score', 0),
        'balance_drain_ratio': num(row, 'balance_drain_ratio', 0),
        'latency_ms': (time.perf_counter() - started) * 1000,
    }


# =============================================================================
# 03. Header
# =============================================================================
st.title('🛡️ FDS V6 교육용 Hybrid MVP')
st.caption(
    '최종 Test 전체 성능 비교 + 저장된 Feature 기반 거래 Serving Demo · '
    '03에서 확정된 ML + Rule Hybrid 설정을 그대로 사용합니다.'
)

st.info('교육용 MVP: 이 화면은 Test에서 Feature Engineering이 끝난 대표 거래를 한 건씩 재생합니다. 실제 운영에서는 신규 원시 거래에 대해 Online Feature Engineering과 Rule Score 계산 계층이 추가로 필요합니다.')
if version_mismatch:
    st.warning('학습 환경과 현재 Serving 환경의 라이브러리 버전이 다릅니다. 03 모델개발 노트북이 생성한 requirements.streamlit.txt로 환경을 맞추는 것을 권장합니다.')

top1, top2, top3, top4 = st.columns(4)
top1.metric('최종 모델', str(ml_meta.get('model_name', '-')))
top2.metric('ML Weight', f'{ML_WEIGHT:.2f}')
top3.metric('Rule Weight', f'{RULE_WEIGHT:.2f}')
top4.metric('Hybrid Threshold', f'{HYBRID_THRESHOLD:.4f}')


# =============================================================================
# 04. Tabs
# =============================================================================
tab_overall, tab_live, tab_log, tab_settings = st.tabs(
    ['📊 전체 Test 성능', '⚡ 교육용 거래 Serving', '📋 처리 로그', '⚙️ 모델/환경 설정']
)


# =============================================================================
# TAB 1. 최종 Test 전체 성능
# =============================================================================
with tab_overall:
    st.subheader('최종 Test 전체: 실제 사기 vs Hybrid 검출 결과')
    st.caption(
        '모델 선택과 Threshold 조정에 사용하지 않은 최종 Test 전체를 기준으로 표시합니다. '
        'Demo Sample이 아니라 Test 전체 집계입니다.'
    )

    summary = hybrid_meta.get('test_summary')
    comparison = pd.DataFrame(hybrid_meta.get('test_comparison', []))
    by_source = pd.DataFrame(hybrid_meta.get('test_by_source', []))
    by_fraud_type = pd.DataFrame(hybrid_meta.get('test_by_fraud_type', []))

    if summary:
        total_events = int(summary.get('total_events', 0))
        actual_fraud = int(summary.get('actual_fraud', 0))
        predicted_fraud = int(summary.get('predicted_fraud', 0))
        tp = int(summary.get('true_positive', 0))
        fp = int(summary.get('false_positive', 0))
        fn = int(summary.get('false_negative', 0))
        tn = int(summary.get('true_negative', 0))

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric('전체 Test 거래', f'{total_events:,}')
        r2.metric('실제 사기', f'{actual_fraud:,}')
        r3.metric('정확히 검출한 사기 (TP)', f'{tp:,}', delta=f'{summary.get("recall", 0):.1%} Recall')
        r4.metric('놓친 사기 (FN)', f'{fn:,}')
        r5.metric('정상 오탐 (FP)', f'{fp:,}')

        st.markdown(
            f"""
            <div class="fds-card">
                <div class="fds-title">핵심 비교</div>
                <div class="fds-sub">
                    실제 사기 <b>{actual_fraud:,}건</b> 중
                    Hybrid가 정확히 잡은 사기는 <span class="good">{tp:,}건</span>,
                    놓친 사기는 <span class="bad">{fn:,}건</span>입니다.
                    Hybrid가 사기로 Alert한 전체 건수는 <b>{predicted_fraud:,}건</b>이며,
                    그중 정상거래 오탐은 <span class="bad">{fp:,}건</span>입니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1])

        with left:
            st.markdown('#### 실제 사기와 검출 결과')
            fraud_compare = pd.DataFrame(
                {
                    '구분': ['실제 사기', '정확히 검출(TP)', '놓친 사기(FN)', '사기 Alert 전체'],
                    '건수': [actual_fraud, tp, fn, predicted_fraud],
                }
            ).set_index('구분')

            st.bar_chart(fraud_compare, y='건수', horizontal=True)

        with right:
            st.markdown('#### Confusion Matrix')
            cm = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=['실제 정상', '실제 사기'],
                columns=['예측 정상', '예측 사기'],
            )
            st.dataframe(
                cm.style.format('{:,.0f}'),
                use_container_width=True,
                height=150,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric('Precision', f'{summary.get("precision", 0):.2%}')
            m2.metric('Recall', f'{summary.get("recall", 0):.2%}')
            m3.metric('F2', f'{summary.get("f2", 0):.4f}')

        if not comparison.empty:
            st.markdown('#### Rule / ML / Hybrid 최종 Test 비교')

            show_cols = [
                'method', 'actual_fraud', 'predicted_fraud',
                'true_positive', 'false_positive', 'false_negative',
                'precision', 'recall', 'f1', 'f2', 'roc_auc', 'pr_auc',
            ]
            show_cols = [c for c in show_cols if c in comparison.columns]

            comparison_view = comparison[show_cols].copy()

            pct_cols = [c for c in ['precision', 'recall', 'f1', 'f2', 'roc_auc', 'pr_auc'] if c in comparison_view.columns]
            count_cols = [c for c in ['actual_fraud', 'predicted_fraud', 'true_positive', 'false_positive', 'false_negative'] if c in comparison_view.columns]

            st.dataframe(
                comparison_view.style
                .format({**{c: '{:,.0f}' for c in count_cols}, **{c: '{:.4f}' for c in pct_cols}}),
                use_container_width=True,
                hide_index=True,
            )

        if not by_source.empty:
            st.markdown('#### 거래 원천별 Hybrid 성능')
            source_view = by_source.copy()

            st.dataframe(
                source_view.style.format(
                    {
                        'total_events': '{:,.0f}',
                        'actual_fraud': '{:,.0f}',
                        'predicted_fraud': '{:,.0f}',
                        'true_positive': '{:,.0f}',
                        'false_positive': '{:,.0f}',
                        'false_negative': '{:,.0f}',
                        'true_negative': '{:,.0f}',
                        'precision': '{:.2%}',
                        'recall': '{:.2%}',
                        'alert_rate': '{:.4%}',
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        if not by_fraud_type.empty:
            st.markdown('#### Fraud Type별 검출률')
            fraud_type_view = by_fraud_type.copy()

            st.dataframe(
                fraud_type_view.style.format(
                    {
                        'actual_fraud': '{:,.0f}',
                        'detected_fraud': '{:,.0f}',
                        'missed_fraud': '{:,.0f}',
                        'avg_hybrid_score': '{:.4f}',
                        'recall': '{:.2%}',
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        test_start = summary.get('test_start_time')
        test_end = summary.get('test_end_time')
        if test_start or test_end:
            st.caption(f'Test 평가기간: {test_start} ~ {test_end}')

    else:
        st.warning(
            '현재 fds_hybrid_metadata.json에는 전체 Test 건수/Confusion Matrix 요약이 없습니다. '
            '함께 제공한 01~03 Notebook을 마지막 셀까지 다시 실행하면 이 화면이 자동으로 채워집니다.'
        )

        if not comparison.empty:
            st.markdown('#### 현재 Metadata에서 확인 가능한 Test 성능')
            st.dataframe(comparison, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 2. 실시간 거래 시연
# =============================================================================
with tab_live:
    st.subheader('교육용 거래 Serving Demo')
    st.caption('Test에서 만든 대표 Feature Row를 한 건씩 또는 Batch로 재생해 저장 모델의 Serving 흐름을 확인합니다.')

    if 'stream_index' not in st.session_state:
        st.session_state.stream_index = 0

    if 'event_log' not in st.session_state:
        st.session_state.event_log = []

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button('거래 1건 처리', use_container_width=True, type='primary'):
            row = events.iloc[st.session_state.stream_index % len(events)]
            st.session_state.event_log.append(predict_event(row))
            st.session_state.stream_index += 1

    with c2:
        batch_size = st.selectbox('Batch 건수', [5, 10, 20], index=1)

    with c3:
        b1, b2 = st.columns(2)

        with b1:
            if st.button('Batch 재생', use_container_width=True):
                for _ in range(batch_size):
                    row = events.iloc[st.session_state.stream_index % len(events)]
                    st.session_state.event_log.append(predict_event(row))
                    st.session_state.stream_index += 1

        with b2:
            if st.button('초기화', use_container_width=True):
                st.session_state.stream_index = 0
                st.session_state.event_log = []
                st.rerun()

    log = pd.DataFrame(st.session_state.event_log[-500:])

    if log.empty:
        st.info('거래를 처리하면 이 영역에 Hybrid 판단 결과가 표시됩니다.')
    else:
        latest = log.iloc[-1]

        q1, q2, q3, q4 = st.columns(4)
        q1.metric('처리 Event', f'{len(log):,}')
        q2.metric('Alert', f'{int(log["alert"].sum()):,}')
        q3.metric('평균 ML Risk', f'{log["ml_risk_score"].mean():.2%}')
        q4.metric('평균 추론시간', f'{log["latency_ms"].mean():.1f} ms')

        st.markdown('#### 가장 최근 거래')

        action = latest['action']
        if action == 'BLOCK_AND_REVIEW':
            st.error('권고 조치: BLOCK AND REVIEW')
        elif action == 'TEMP_HOLD':
            st.error('권고 조치: TEMP HOLD')
        elif action == 'STEP_UP_AUTH':
            st.warning('권고 조치: STEP UP AUTH')
        else:
            st.success('권고 조치: PASS')

        a1, a2 = st.columns(2)

        with a1:
            st.markdown(
                f"""
                <div class="fds-card">
                    <div class="fds-title">거래 정보</div>
                    <div class="fds-sub">
                        원천: <b>{latest['source_table']}</b><br>
                        거래유형: <b>{latest['event_subtype']}</b><br>
                        거래금액: <b>{latest['amount_abs']:,.0f}원</b><br>
                        Event: <b>{latest['event_uid']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with a2:
            s1, s2, s3 = st.columns(3)
            s1.metric('Rule', f'{latest["rule_score"]:.0f}/100')
            s2.metric('ML Risk', f'{latest["ml_risk_score"]:.2%}')
            s3.metric('Hybrid', f'{latest["hybrid_score"]:.2%}')

        st.markdown('#### 현재 Event 위험신호')

        f1, f2, f3, f4 = st.columns(4)
        f1.metric('신규기기', 'YES' if latest['is_new_device'] else 'NO')
        f2.metric('해외 IP', 'YES' if latest['foreign_ip'] else 'NO')
        f3.metric('신규 수취인', 'YES' if latest['new_beneficiary'] else 'NO')
        f4.metric('수취인 위험점수', f'{latest["beneficiary_risk"]:.2f}')

        drain = float(np.clip(latest['balance_drain_ratio'], 0, 1))
        st.progress(drain, text=f'계좌 잔액 소진율 proxy: {latest["balance_drain_ratio"]:.1%}')

        with st.expander('교육용 실제 정답 확인'):
            if latest['actual_label'] == 1:
                st.error(f'실제 정답: FRAUD / {latest["actual_fraud_type"]}')
            else:
                st.success('실제 정답: NORMAL')


# =============================================================================
# TAB 3. 처리 로그
# =============================================================================
with tab_log:
    st.subheader('실시간 Demo 처리 로그')

    log = pd.DataFrame(st.session_state.get('event_log', [])[-500:])

    if log.empty:
        st.info('아직 처리한 Demo 거래가 없습니다.')
    else:
        display_cols = [
            'event_at', 'source_table', 'event_subtype', 'amount_abs',
            'rule_score', 'ml_risk_score', 'hybrid_score',
            'alert', 'action', 'actual_label', 'actual_fraud_type', 'latency_ms',
        ]
        display_cols = [c for c in display_cols if c in log.columns]

        st.dataframe(
            log[display_cols].sort_index(ascending=False),
            use_container_width=True,
            hide_index=True,
            height=560,
        )


# =============================================================================
# TAB 4. 모델 설정
# =============================================================================
with tab_settings:
    st.subheader('Serving에 고정된 최종 설정')

    s1, s2 = st.columns(2)

    with s1:
        st.markdown('#### 모델')
        st.json({'model_name': ml_meta.get('model_name'), 'ml_threshold': ml_meta.get('ml_threshold'), 'model_feature_count': len(MODEL_FEATURES), 'numeric_feature_count': len(NUMERIC_FEATURES), 'categorical_feature_count': len(CATEGORICAL_FEATURES), 'train_end_time': ml_meta.get('train_end_time'), 'valid_end_time': ml_meta.get('valid_end_time'), 'model_sha256_verified': True})

    with s2:
        st.markdown('#### Hybrid / Action')
        st.json({'ml_weight': ML_WEIGHT, 'rule_weight': RULE_WEIGHT, 'hybrid_threshold': HYBRID_THRESHOLD, 'fraud_action_thresholds': FRAUD_ACTION_THRESHOLDS, 'selection_data': hybrid_meta.get('selection_data'), 'test_usage': hybrid_meta.get('test_usage')})

    st.markdown('#### 학습 환경 ↔ Serving 환경 버전 확인')
    st.dataframe(version_check, use_container_width=True, hide_index=True)
    if version_mismatch:
        st.warning('CHECK 항목이 있습니다. 특히 scikit-learn / xgboost / lightgbm 버전 차이는 joblib 모델 로딩 또는 예측 결과에 영향을 줄 수 있습니다.')
    else:
        st.success('모델 학습 시 기록된 핵심 Runtime 버전과 현재 Serving 환경이 일치합니다.')

    st.markdown('#### Rule Config')
    st.caption('현재 교육용 Demo의 rule_score는 03에서 미리 계산되어 저장됩니다. Rule Config는 감사/확인용으로 함께 보관하며, 실제 운영에서는 신규 원시 거래에서 Rule Score를 계산할 때 사용합니다.')
    st.json({'rule_count': len(rule_meta.get('rule_config', {})), 'rule_score_max': rule_meta.get('rule_score_max'), 'final_rule_threshold': rule_meta.get('final_rule_threshold')})

    with st.expander('실제 운영 FDS로 확장할 때 필요한 계층'):
        st.code('신규 원시 거래\n→ Online Feature Engineering / Feature Store\n→ History Feature + Rule Feature 계산\n→ Rule Score\n→ 저장된 ML Pipeline predict_proba()\n→ Hybrid Score\n→ PASS / STEP_UP_AUTH / TEMP_HOLD / BLOCK_AND_REVIEW')

    st.caption('이 화면은 교육용 Model Serving MVP입니다. 모델 학습, 후보 모델 비교, Threshold 재탐색은 수행하지 않습니다.')
