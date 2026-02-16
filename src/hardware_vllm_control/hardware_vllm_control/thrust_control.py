#!/usr/bin/env python3
import time
import pigpio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

# GPIO 번호 (BCM)
LEFT_GPIO = 13   # 좌측 모터
RIGHT_GPIO = 18  # 우측 모터

# ESC/서보용 펄스폭 (마이크로초 단위)
PULSE_FORWARD_US = 1000   # forward (1.0ms)
PULSE_STOP_US    = 1500   # neutral/stop (1.5ms)
PULSE_BACK_US    = 2000   # backward (2.0ms)

# 명령 유지 시간 (초)
MOTION_DURATION_SEC = 2.0

# ===== PWM 램프(soft start/stop) 설정 =====
RAMP_TIME_SEC = 0.5   # 목표 PWM 도달 시간 (0.5초 이내)
RAMP_HZ = 50          # 램프 갱신 주기 (Hz) - 20~50 권장
RAMP_DT = 1.0 / RAMP_HZ


class MotorPwmNode(Node):
    def __init__(self):
        super().__init__('motor_pwm_node')

        # pigpio 연결 (localhost)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon에 연결할 수 없습니다. 'sudo pigpiod' 실행 여부 확인")

        # ESC 중립(정지)로 초기화 + arming 시간
        self.get_logger().info("초기 중립 신호(1500µs)를 양쪽 모터에 출력합니다 (arming).")
        self.set_both_pwm(PULSE_STOP_US)
        time.sleep(1.0)

        # ===== thrust_busy publish =====
        # 절대경로 토픽 추천: /thrust_busy
        self.busy_pub = self.create_publisher(Bool, '/thrust_busy', 10)
        self.is_busy = False
        self.publish_busy(False)

        # /cmd_motor 토픽 구독
        self.sub = self.create_subscription(
            String,
            '/cmd_motor',
            self.cmd_callback,
            10
        )

        # 자동 정지 타이머 핸들
        self.stop_timer = None

        # ===== 램프 상태 =====
        self.left_pulse = PULSE_STOP_US
        self.right_pulse = PULSE_STOP_US
        self.left_target = PULSE_STOP_US
        self.right_target = PULSE_STOP_US
        self.ramp_timer = None

        self.get_logger().info(
            "MotorPwmNode(pigpio) 시작.\n"
            "/cmd_motor 토픽에 forward/backward/left/right/stop 문자열을 publish 하면,\n"
            f"{MOTION_DURATION_SEC}초 동안 동작 후 자동으로 정지(1500µs)합니다.\n"
            f"PWM 램프: {RAMP_TIME_SEC}s 이내 목표 도달, {RAMP_HZ}Hz 업데이트"
        )

    # pigpio 서보 펄스 설정 (µs 단위)
    def set_pwm(self, gpio: int, pulse_us: int):
        pulse_us = max(500, min(2500, int(pulse_us)))
        self.pi.set_servo_pulsewidth(gpio, pulse_us)

    def set_both_pwm(self, pulse_us: int):
        self.set_pwm(LEFT_GPIO, pulse_us)
        self.set_pwm(RIGHT_GPIO, pulse_us)

    def publish_busy(self, state: bool):
        self.is_busy = bool(state)
        self.busy_pub.publish(Bool(data=self.is_busy))

    # ===== 램프(soft start/stop) 구현 =====
    def _start_ramp_to(self, left_target: int, right_target: int):
        self.left_target = int(left_target)
        self.right_target = int(right_target)

        # 기존 램프 타이머 취소
        if self.ramp_timer is not None:
            self.ramp_timer.cancel()
            self.ramp_timer = None

        # 동작 시작으로 간주 → busy True
        self.publish_busy(True)

        # 램프 시작
        self.ramp_timer = self.create_timer(RAMP_DT, self._ramp_step)

    def _ramp_step(self):
        steps = max(1, int(RAMP_TIME_SEC / RAMP_DT))

        def step_toward(cur: int, tgt: int) -> int:
            if cur == tgt:
                return cur
            delta = tgt - cur
            inc = int(round(delta / steps))
            if inc == 0:
                inc = 1 if delta > 0 else -1
            nxt = cur + inc
            # overshoot 방지
            if (delta > 0 and nxt > tgt) or (delta < 0 and nxt < tgt):
                nxt = tgt
            return nxt

        self.left_pulse = step_toward(self.left_pulse, self.left_target)
        self.right_pulse = step_toward(self.right_pulse, self.right_target)

        self.set_pwm(LEFT_GPIO, self.left_pulse)
        self.set_pwm(RIGHT_GPIO, self.right_pulse)

        # 목표 도달 시 램프 종료
        if self.left_pulse == self.left_target and self.right_pulse == self.right_target:
            if self.ramp_timer is not None:
                self.ramp_timer.cancel()
                self.ramp_timer = None

            # 목표가 중립이면 busy False
            if self.left_target == PULSE_STOP_US and self.right_target == PULSE_STOP_US:
                self.publish_busy(False)

    # ===== 모션 명령 + 자동 정지 =====
    def start_motion_with_auto_stop(self, motion_desc: str, left_pulse_us: int, right_pulse_us: int):
        self.get_logger().info(
            f"명령: {motion_desc} (좌 {left_pulse_us}µs / 우 {right_pulse_us}µs) "
            f"-> {RAMP_TIME_SEC}s 램프로 목표 도달"
        )

        # busy 중이면 새 명령 거부(단, stop은 cmd_callback에서 예외)
        if self.is_busy:
            self.get_logger().warn(f"BUSY: '{motion_desc}' 명령 무시 (동작 중)")
            return

        # 목표로 램프 이동 시작
        self._start_ramp_to(left_pulse_us, right_pulse_us)

        # 기존 자동 정지 타이머가 있으면 취소
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        # MOTION_DURATION_SEC 후 중립으로 램프 다운
        self.stop_timer = self.create_timer(
            MOTION_DURATION_SEC,
            self.auto_stop_callback
        )

    def auto_stop_callback(self):
        # 타이머가 주기적으로 돌지 않도록 바로 취소
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        self.get_logger().info(f"{MOTION_DURATION_SEC}초 경과: 자동 정지(중립)로 램프 다운")
        self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US)
        # busy False는 램프가 중립 도달 시 _ramp_step에서 처리

    def cmd_callback(self, msg: String):
        cmd = msg.data.strip().lower()

        # 모터 동작 중이면 새 명령 무시 (단, stop은 즉시 허용)
        if self.is_busy and cmd != 'stop':
            self.get_logger().warn(f"BUSY: '{cmd}' 명령 무시 (동작 중)")
            return

        if cmd == 'forward':
            self.start_motion_with_auto_stop("FORWARD", PULSE_FORWARD_US, PULSE_FORWARD_US)

        elif cmd == 'backward':
            self.start_motion_with_auto_stop("BACKWARD", PULSE_BACK_US, PULSE_BACK_US)

        elif cmd == 'left':
            self.start_motion_with_auto_stop("LEFT", PULSE_BACK_US, PULSE_FORWARD_US)

        elif cmd == 'right':
            self.start_motion_with_auto_stop("RIGHT", PULSE_FORWARD_US, PULSE_BACK_US)

        elif cmd == 'stop':
            # STOP은 즉시 적용하되, PWM 점프가 싫다면 램프 다운으로 처리
            self.get_logger().info("명령: STOP (중립으로 램프 다운)")

            # 기존 자동 정지 타이머 취소
            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None

            # 램프 중이라면 계속 중립으로 유도
            self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US)

        else:
            self.get_logger().warn(
                f"알 수 없는 명령: '{cmd}'. "
                "사용 가능: forward / backward / left / right / stop"
            )

    def cleanup(self):
        self.get_logger().info("노드 종료: 모터 정지 및 pigpio 해제.")

        # 타이머들 정리
        try:
            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None
            if self.ramp_timer is not None:
                self.ramp_timer.cancel()
                self.ramp_timer = None
        except Exception:
            pass

        try:
            # busy 해제 공지(가능하면 먼저)
            self.publish_busy(False)

            # 정지 신호(중립) 후 PWM off
            self.set_both_pwm(PULSE_STOP_US)
            time.sleep(0.3)
            self.pi.set_servo_pulsewidth(LEFT_GPIO, 0)
            self.pi.set_servo_pulsewidth(RIGHT_GPIO, 0)
        except Exception as e:
            self.get_logger().error(f"GPIO 정리 중 에러: {e}")
        finally:
            self.pi.stop()


def main(args=None):
    rclpy.init(args=args)
    node = MotorPwmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.cleanup()
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
