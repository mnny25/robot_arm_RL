"""Push v2 실험 설정 모음

각 설정은
    "env": PushEnvV2에 넘길 설정 (push_env_v2.DEFAULT_CONFIG의 키만 사용)
    "curriculum": True면 학습 중 range_scale을 0 → 1로 점점 키움
으로 구성됩니다. 새 실험은 여기에 이름을 하나 추가하고 --config 이름으로 실행하세요.
"""

FULL_DR = {
    "friction": (0.5, 1.5),
    "mass": (0.03, 0.08),
    "kp_scale": (0.8, 1.2),
    "damping_scale": (0.8, 1.2),
    "obs_noise_std": 0.003,       # 큐브 위치 관측 노이즈 3mm
    "cube_pos_bias_std": 0.003,   # 에피소드마다 고정된 카메라 오차 3mm
    "action_delay_max": 2,        # 0~2 스텝(0~0.1초) 지연
}

CONFIGS = {
    # ---- Phase 1: 성공 판정(hold) 효과 확인, 위치 고정 ----
    # 기존 실험 재현용: 원에 들어가는 순간 성공, 속도 페널티 없음
    "fixed_nohold": {
        "env": {
            "range_scale": 0.0,
            "require_stable": False,
            "w_cube_speed": 0.0,
            "w_zone_speed": 0.0,
        },
        "curriculum": False,
    },
    # 0.5초 정지 유지 + 속도 페널티
    "fixed_hold": {
        "env": {"range_scale": 0.0},
        "curriculum": False,
    },

    # ---- Phase 2: 큐브·목표 랜덤화 ----
    "rand": {
        "env": {"range_scale": 1.0},
        "curriculum": True,
    },
    # 커리큘럼 효과 비교용 (처음부터 전체 범위)
    "rand_nocurr": {
        "env": {"range_scale": 1.0},
        "curriculum": False,
    },

    # ---- Phase 3: Domain Randomization ----
    "rand_dr_friction": {
        "env": {"range_scale": 1.0, "dr": {"friction": (0.5, 1.5)}},
        "curriculum": True,
    },
    "rand_dr_full": {
        "env": {"range_scale": 1.0, "dr": FULL_DR},
        "curriculum": True,
    },
    # 실물에서 얻기 쉬운 관측만 사용 (큐브 자세·속도 제외)
    "rand_dr_full_real": {
        "env": {"range_scale": 1.0, "dr": FULL_DR, "obs_mode": "real"},
        "curriculum": True,
    },
}


def get_config(name: str) -> dict:
    if name not in CONFIGS:
        raise KeyError(f"설정 '{name}' 없음. 사용 가능: {', '.join(CONFIGS)}")
    return CONFIGS[name]
