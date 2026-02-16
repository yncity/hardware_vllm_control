#!/usr/bin/env python3
import time
import pigpio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

# GPIO 번호 (BCM)
LEFT_GPIO = 13
RIGHT_GPIO = 18

# ESC/서보용 펄스폭 (마이크로초 단위)
PULSE_FORWARD_US = 1000
PULSE_STOP_US    = 1500
PULSE_BACK_US    = 2000

# 명령 유지 시간 (초)
MOTION_DURATION_SEC = 2.0

# ===== PWM 램프(soft start/stop) 설정 =====
RAMP_TIME_SEC = 0.5   # 목표 PWM 도달 시간
RAMP_HZ = 50          # 램프 갱신 주기 (Hz)
RAMP_DT = 1.0 / RAMP_HZ


class MotorPwmNode(Node):
    def __init__(self):
        super().__init__('motor_pwm_node')

        # pigpio 연결 (localhost)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon에 연결할 수 없습니다. 'sudo pigpiod' 실행 여부 확인")

        # ESC arming: 중립 유지
        self.get_logger().info("초기 중립 신호(1500µs)를 양쪽 모터에 출력합니다 (arming).")
        self.set_both_pwm(PULSE_STOP_US)
        time.sleep(1.0)

        # thrust_busy publish
        self.busy_pub = self.create_publisher(Bool, '/thrust_busy', 10)
        self.is_busy = False
        self.publish_busy(False)

        # /cmd_motor 구독
        self.sub = self.create_subscription(String, '/cmd_motor', self.cmd_callback, 10)

        # 자동 정지 타이머
        self.stop_timer = None

        # ===== S-curve 램프 상태 =====
        self.left_pulse = PULSE_STOP_US
        self.right_pulse = PULSE_STOP_US
        self.left_start = PULSE_STOP_US
        self.right_start = PULSE_STOP_US
        self.left_target = PULSE_STOP_US
        self.right_target = PULSE_STOP_US

        self.ramp_timer = None
        self.ramp_t0 = 0.0
        self.ramp_duration = RAMP_TIME_SEC

        self.get_logger().info(
            "MotorPwmNode(pigpio) 시작.\n"
            "/cmd_motor: forward/backward/left/right/stop\n"
            f"램프: S-curve(3t^2-2t^3), {RAMP_TIME_SEC}s, {RAMP_HZ}Hz\n"
            f"자동정지: {MOTION_DURATION_SEC}s 후 중립"
        )

    def set_pwm(self, gpio: int, pulse_us: int):
        pulse_us = max(500, min(2500, int(pulse_us)))
        self.pi.set_servo_pulsewidth(gpio, pulse_us)

    def set_both_pwm(self, pulse_us: int):
        self.set_pwm(LEFT_GPIO, pulse_us)
        self.set_pwm(RIGHT_GPIO, pulse_us)

    def publish_busy(self, state: bool):
        self.is_busy = bool(state)
        self.busy_pub.publish(Bool(data=self.is_busy))

    # ===== S-curve easing: smoothstep(0->1) =====
    @staticmethod
    def _smoothstep(u: float) -> float:
        # clamp
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        # 3u^2 - 2u^3
        return (3.0 * u * u) - (2.0 * u * u * u)

    def _start_ramp_to(self, left_target: int, right_target: int, duration: float = RAMP_TIME_SEC):
        # 목표 설정
        self.left_target = int(left_target)
        self.right_target = int(right_target)

        # 시작값 저장 (현재 출력값 기준)
        self.left_start = int(self.left_pulse)
        self.right_start = int(self.right_pulse)

        self.ramp_t0 = time.monotonic()
        self.ramp_duration = max(0.01, float(duration))

        # 기존 램프 타이머 취소
        if self.ramp_timer is not None:
            self.ramp_timer.cancel()
            self.ramp_timer = None

        # 램프 구간 동안 busy True
        self.publish_busy(True)

        # 램프 타이머 시작
        self.ramp_timer = self.create_timer(RAMP_DT, self._ramp_step)

    def _ramp_step(self):
        t = time.monotonic() - self.ramp_t0
        u = t / self.ramp_duration
        s = self._smoothstep(u)

        # 보간(interpolation)
        self.left_pulse = int(round(self.left_start + (self.left_target - self.left_start) * s))
        self.right_pulse = int(round(self.right_start + (self.right_target - self.right_start) * s))

        self.set_pwm(LEFT_GPIO, self.left_pulse)
        self.set_pwm(RIGHT_GPIO, self.right_pulse)

        # 완료 조건
        if u >= 1.0:
            # 정확히 목표값으로 고정
            self.left_pulse = self.left_target
            self.right_pulse = self.right_target
            self.set_pwm(LEFT_GPIO, self.left_pulse)
            self.set_pwm(RIGHT_GPIO, self.right_pulse)

            if self.ramp_timer is not None:
                self.ramp_timer.cancel()
                self.ramp_timer = None

            # 목표가 중립이면 busy False
            if self.left_target == PULSE_STOP_US and self.right_target == PULSE_STOP_US:
                self.publish_busy(False)

    def start_motion_with_auto_stop(self, motion_desc: str, left_pulse_us: int, right_pulse_us: int):
        # busy 중이면 새 명령 거부(단, stop은 cmd_callback에서 예외)
        if self.is_busy:
            self.get_logger().warn(f"BUSY: '{motion_desc}' 명령 무시 (동작 중)")
            return

        self.get_logger().info(
            f"명령: {motion_desc} (좌 {left_pulse_us}µs / 우 {right_pulse_us}µs) "
            f"-> S-curve 램프 {RAMP_TIME_SEC}s"
        )

        # 목표로 S-curve 램프
        self._start_ramp_to(left_pulse_us, right_pulse_us, duration=RAMP_TIME_SEC)

        # 기존 자동 정지 타이머 취소
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        # MOTION_DURATION_SEC 후 중립으로 램프 다운
        self.stop_timer = self.create_timer(MOTION_DURATION_SEC, self.auto_stop_callback)

    def auto_stop_callback(self):
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        self.get_logger().info(f"{MOTION_DURATION_SEC}초 경과: 자동 정지(중립)로 S-curve 램프 다운")
        self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US, duration=RAMP_TIME_SEC)

    def cmd_callback(self, msg: String):
        cmd = msg.data.strip().lower()

        # 동작 중이면 새 명령 무시 (단, stop은 허용)
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
            self.get_logger().info("명령: STOP (중립으로 S-curve 램프 다운)")

            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None

            # stop도 부드럽게: 중립으로 램프
            self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US, duration=RAMP_TIME_SEC)

        else:
            self.get_logger().warn(
                f"알 수 없는 명령: '{cmd}'. "
                "사용 가능: forward / backward / left / right / stop"
            )

    def cleanup(self):
        self.get_logger().info("노드 종료: 모터 정지 및 pigpio 해제.")

        # 타이머 정리
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
            # 먼저 busy 해제
            self.publish_busy(False)

            # 중립 출력 후 PWM off
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
