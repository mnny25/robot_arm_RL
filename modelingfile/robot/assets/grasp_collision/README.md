# SO-101 집게 충돌 형상

출처: MuJoCo Menagerie의 robotstudio_so101 모델과 자산.
https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101

라이선스: 함께 보관한 LICENSE의 Apache License 2.0.

so101_grasp.xml은 기존 보정 모델의 시각 형상, 관절, 액추에이터 및 관성 설정을 유지합니다.
고정 집게와 이동 집게의 통짜 충돌 메시만 분할 메시와 기본 충돌 형상으로 교체했습니다.
접촉 파라미터는 기존 collision 클래스를 사용하며, 가져온 형상은 질량 계산에서 제외합니다.
원본 모델과 Push 환경은 변경하지 않습니다.

scene_grasp.xml은 파지 유지 중 수치적 미끄러짐을 줄이기 위해 cone="elliptic", impratio="10"을 사용합니다.
질량, 관성, 관절 범위, 액추에이터 설정, 적분 방식 및 시뮬레이션 시간 간격은 원본과 같습니다.
