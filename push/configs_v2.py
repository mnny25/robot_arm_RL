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

# 실측값을 얻기 전의 시작 범위. 전체 DR과 같은 항목을 좁은 범위에서 적용한다.
REAL_DR_MILD = {
    "friction": (0.8, 1.2),
    "mass": (0.04, 0.06),
    "kp_scale": (0.9, 1.1),
    "damping_scale": (0.9, 1.1),
    "obs_noise_std": 0.001,
    "cube_pos_bias_std": 0.001,
    "action_delay_max": 1,
}

# EE를 '밀 위치'(큐브의 목표 반대편)로 유도하는 보상 설정
PP = {"ee_target": "push_point", "w_ee": 2.0}

# 로봇(집게)이 테이블에 닿거나 누르면 페널티 (Sim-to-Real: 테이블 긁기 방지)
# 접촉 스텝당 0.5 + 접촉 힘 비례 최대 0.5 → 다른 항(스텝당 약 -0.5~-1)과 비슷한 크기
CONTACT = {"w_table_contact": 0.5, "w_table_force": 0.5, "table_force_cap": 20.0}

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

    # 0.5초 정지 유지 + EE를 '밀 위치'(큐브의 목표 반대편)로 유도
    "fixed_hold_pp": {
        "env": {"range_scale": 0.0, "ee_target": "push_point", "w_ee": 2.0},
        "curriculum": False,
    },

    # ---- Phase 2: 큐브·목표 랜덤화 ----
    "rand_pp": {
        "env": {"range_scale": 1.0, "ee_target": "push_point", "w_ee": 2.0},
        "curriculum": True,
    },
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

    # ---- Phase 3 (push point 버전): Phase 2의 rand_pp가 비교 기준 ----
    "rand_pp_dr_friction": {
        "env": {**PP, "range_scale": 1.0, "dr": {"friction": (0.5, 1.5)}},
        "curriculum": True,
    },
    "rand_pp_dr_full": {
        "env": {**PP, "range_scale": 1.0, "dr": FULL_DR},
        "curriculum": True,
    },
    "rand_pp_dr_full_real": {
        "env": {**PP, "range_scale": 1.0, "dr": FULL_DR, "obs_mode": "real"},
        "curriculum": True,
    },

    # ---- Phase 4: 테이블 접촉 페널티 (DR 없음, rand_pp가 비교 기준) ----
    "rand_pp_contact": {
        "env": {**PP, **CONTACT, "range_scale": 1.0},
        "curriculum": True,
    },
    "rand_pp_contact_full_table": {
        "env": {**PP, **CONTACT, "range_scale": 1.0, "sample_full_table": True},
        "curriculum": True,
    },
    # 접촉 페널티 + 마찰 DR (최종 후보). 근거: SAC는 노이즈·지연·kp에 원래 강건하고 약점은 높은 마찰뿐
    # (민감도: 마찰 1.5 → 0.80, 2.0 → 0.64). rand_pp_contact와 비교하면 마찰 DR 효과만 분리됨
    "rand_pp_contact_drf": {
        "env": {**PP, **CONTACT, "range_scale": 1.0, "dr": {"friction": (0.5, 2.0)}},
        "curriculum": True,
    },
    # 접촉 페널티 + push point를 5mm 올림 (EE가 테이블에서 떨어진 채 밀도록)
    "rand_pp_contact_up": {
        "env": {**PP, **CONTACT, "range_scale": 1.0, "push_point_height": 0.005},
        "curriculum": True,
    },

    # ---- Phase 3 원인 찾기 (ablation): FULL_DR의 요소를 하나씩만 켬 ----
    "rand_pp_dr_delay": {
        "env": {**PP, "range_scale": 1.0, "dr": {"action_delay_max": 2}},
        "curriculum": True,
    },
    "rand_pp_dr_obs": {
        "env": {**PP, "range_scale": 1.0,
                "dr": {"obs_noise_std": 0.003, "cube_pos_bias_std": 0.003}},
        "curriculum": True,
    },
    "rand_pp_dr_dyn": {
        "env": {**PP, "range_scale": 1.0,
                "dr": {"kp_scale": (0.8, 1.2), "damping_scale": (0.8, 1.2), "mass": (0.03, 0.08)}},
        "curriculum": True,
    },
}

# contact_drf와 같은 조건에서 학습 에피소드의 25%를 복구 자세에서 시작한다.
CONFIGS["rand_pp_contact_drf_recovery"] = {
    **CONFIGS["rand_pp_contact_drf"],
    "env": {**CONFIGS["rand_pp_contact_drf"]["env"], "recovery_start_prob": 0.25},
}

# 배치만 넓혀 contact_drf와 비교한다.
CONFIGS["rand_pp_contact_drf_full_table"] = {
    **CONFIGS["rand_pp_contact_drf"],
    "env": {
        **CONFIGS["rand_pp_contact_drf"]["env"],
        "sample_full_table": True,
        "require_push_reach": True,
    },
}

# 실물 관측으로 학습할 때 위치·보상은 고정하고 DR 적용 여부만 비교한다.
CONFIGS["rand_pp_contact_real"] = {
    "env": {**PP, **CONTACT, "range_scale": 1.0, "obs_mode": "real"},
    "curriculum": True,
}
CONFIGS["rand_pp_contact_real_dr_mild"] = {
    **CONFIGS["rand_pp_contact_real"],
    "env": {**CONFIGS["rand_pp_contact_real"]["env"], "dr": REAL_DR_MILD},
}


def get_config(name: str) -> dict:
    if name not in CONFIGS:
        raise KeyError(f"설정 '{name}' 없음. 사용 가능: {', '.join(CONFIGS)}")
    return CONFIGS[name]
