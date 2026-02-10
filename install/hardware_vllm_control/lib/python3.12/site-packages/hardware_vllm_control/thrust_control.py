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
PULSE_FORWARD_US = 1000   # 앞 (1.0ms)
PULSE_STOP_US    = 1500   # 정지(중립, 1.5ms)
PULSE_BACK_US    = 2000   # 뒤 (2.0ms)

# 명령 유지 시간 (초) — 여기 숫자만 바꾸면 2초 → 3초 등 조절 가능
MOTION_DURATION_SEC = 2.0


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
        time.sleep(1.0)  # ESC에 맞게 필요하면 늘려도 됨

        # busy 상태 publish (모터 동작 중이면 True)
        self.busy_pub = self.create_publisher(Bool, '/thrust_busy', 10)
        self.is_busy = False
        self.publish_busy(False)  # 초기값

        # /cmd_motor 토픽 구독
        self.sub = self.create_subscription(
            String,
            '/cmd_motor',
            self.cmd_callback,
            10
        )

        # 자동 정지 타이머 핸들
        self.stop_timer = None

        self.get_logger().info(
            "MotorPwmNode(pigpio) 시작.\n"
            "/cmd_motor 토픽에 forward/backward/left/right/stop 문자열을 publish 하면,\n"
            f"{MOTION_DURATION_SEC}초 동안 동작 후 자동으로 정지(1500µs)합니다."
        )

    # pigpio 서보 펄스 설정 (µs 단위)
    def set_pwm(self, gpio: int, pulse_us: int):
        # 안전 범위 (pigpio는 보통 500~2500µs 권장)
        pulse_us = max(500, min(2500, pulse_us))
        self.pi.set_servo_pulsewidth(gpio, pulse_us)

    def publish_busy(self, state: bool):
        self.is_busy = state
        self.busy_pub.publish(Bool(data=state))


    def set_both_pwm(self, pulse_us: int):
        self.set_pwm(LEFT_GPIO, pulse_us)
        self.set_pwm(RIGHT_GPIO, pulse_us)

    def start_motion_with_auto_stop(self, motion_desc: str, left_pulse_us: int, right_pulse_us: int):
        # 모터 명령 적용 + MOTION_DURATION_SEC 후 자동 정지 타이머 설정

        self.get_logger().info(
            f"명령: {motion_desc} (좌 {left_pulse_us}µs / 우 {right_pulse_us}µs)"
        )

        # 즉시 해당 명령 적용
        self.set_pwm(LEFT_GPIO, left_pulse_us)
        self.set_pwm(RIGHT_GPIO, right_pulse_us)

        # 모터 동작 시작 → busy True
        self.publish_busy(True)

        # 기존에 대기 중이던 자동 정지 타이머가 있으면 취소
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        # 새로 한 번만 실행되는 타이머 생성
        self.stop_timer = self.create_timer(
            MOTION_DURATION_SEC,
            self.auto_stop_callback
        )

    def auto_stop_callback(self):
        # MOTION_DURATION_SEC 후 호출 → 양쪽 모터 정지(중립)

        # 타이머가 주기적으로 돌지 않도록 바로 취소
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None
        self.get_logger().info(f"{MOTION_DURATION_SEC}초 경과: 자동 정지 (좌/우 1500µs)")
        self.set_both_pwm(PULSE_STOP_US)

        # 동작 종료 → busy False
        self.publish_busy(False)


    def cmd_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        
        # 모터 동작 중이면 새 명령 무시 (단, stop은 즉시 허용)
        if self.is_busy and cmd != 'stop':
            self.get_logger().warn(f"BUSY: '{cmd}' 명령 무시 (동작 중)")
            return
        
        if cmd == 'forward':
            # 전진: 양쪽 1000µs
            self.start_motion_with_auto_stop("FORWARD", PULSE_FORWARD_US, PULSE_FORWARD_US)

        elif cmd == 'backward':
            # 후진: 양쪽 2000µs
            self.start_motion_with_auto_stop("BACKWARD", PULSE_BACK_US, PULSE_BACK_US)

        elif cmd == 'left':
            # 좌회전: 좌측 뒤(2000µs) / 우측 앞(1000µs)
            self.start_motion_with_auto_stop("LEFT", PULSE_BACK_US, PULSE_FORWARD_US)

        elif cmd == 'right':
            # 우회전: 좌측 앞(1000µs) / 우측 뒤(2000µs)
            self.start_motion_with_auto_stop("RIGHT", PULSE_FORWARD_US, PULSE_BACK_US)

        elif cmd == 'stop':
            self.get_logger().info("명령: STOP (즉시 정지, 좌/우 1500µs)")
            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None
            self.set_both_pwm(PULSE_STOP_US)
            self.publish_busy(False)


        else:
            self.get_logger().warn(
                f"알 수 없는 명령: '{cmd}'. "
                "사용 가능: forward / backward / left / right / stop"
            )

    def cleanup(self):
        self.get_logger().info("노드 종료: 모터 정지 및 pigpio 해제.")
        try:
            # 먼저 busy 해제 공지 (가장 먼저/확실히)
            self.publish_busy(False)

            # 정지 신호
            self.set_both_pwm(PULSE_STOP_US)
            time.sleep(0.5)

            # PWM off
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
    # 이미 shutdown 된 상태면 에러 없이 넘어가도록 방어
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()