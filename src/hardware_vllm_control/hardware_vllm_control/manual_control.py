#!/usr/bin/env python3
import sys
import termios
import tty
import threading
import time

import pigpio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

# =========================
# GPIO (BCM)
# =========================
LEFT_GPIO = 13
RIGHT_GPIO = 12

# =========================
# ESC pulse widths (microseconds)
# =========================
PULSE_FORWARD_US = 1300
PULSE_STOP_US    = 1500
PULSE_BACK_US    = 1700

PULSE_ROTATE_US   = 1300
PULSE_ROTATE_US_2 = 1700

# command hold (sec) then auto stop
MOTION_DURATION_SEC = 1.0

# =========================
# PWM ramp (S-curve)
# =========================
RAMP_TIME_SEC = 0.5
RAMP_HZ = 50
RAMP_DT = 1.0 / RAMP_HZ


class KeyboardMotorPwmNode(Node):
    def __init__(self):
        super().__init__('keyboard_motor_pwm_node')

        # thrust_busy publish (유지)
        self.busy_pub = self.create_publisher(Bool, '/thrust_busy', 10)
        self.is_busy = False
        self.publish_busy(False)

        # pigpio connection
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon 연결 실패. 'sudo pigpiod' 실행 여부 확인")

        # ESC arming
        self.get_logger().info("초기 중립 신호(1500µs)를 양쪽 모터에 출력합니다 (arming).")
        self.set_both_pwm(PULSE_STOP_US)
        time.sleep(1.0)

        # timers
        self.stop_timer = None
        self.ramp_timer = None

        # ramp state
        self.left_pulse = PULSE_STOP_US
        self.right_pulse = PULSE_STOP_US
        self.left_start = PULSE_STOP_US
        self.right_start = PULSE_STOP_US
        self.left_target = PULSE_STOP_US
        self.right_target = PULSE_STOP_US
        self.ramp_t0 = 0.0
        self.ramp_duration = RAMP_TIME_SEC

        # keyboard thread control
        self._running = True
        self._kbd_thread = threading.Thread(target=self.keyboard_loop, daemon=True)

        self.get_logger().info(
            "통합 키보드→PWM 노드 시작.\n"
            "w: forward  | s: backward\n"
            "a: left     | d: right\n"
            "x: stop     | q: 종료\n"
            f"램프: S-curve {RAMP_TIME_SEC}s @ {RAMP_HZ}Hz\n"
            f"자동정지: {MOTION_DURATION_SEC}s 후 stop\n"
            "주의: 이 터미널에서만 키 입력을 받습니다."
        )

        self._kbd_thread.start()

    # -------------------------
    # PWM helpers
    # -------------------------
    def set_pwm(self, gpio: int, pulse_us: int):
        pulse_us = max(500, min(2500, int(pulse_us)))
        self.pi.set_servo_pulsewidth(gpio, pulse_us)

    def set_both_pwm(self, pulse_us: int):
        self.set_pwm(LEFT_GPIO, pulse_us)
        self.set_pwm(RIGHT_GPIO, pulse_us)

    # -------------------------
    # Busy publish
    # -------------------------
    def publish_busy(self, state: bool):
        self.is_busy = bool(state)
        self.busy_pub.publish(Bool(data=self.is_busy))

    # -------------------------
    # Terminal raw key input
    # -------------------------
    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def keyboard_loop(self):
        while rclpy.ok() and self._running:
            key = self.get_key().lower()

            if key == 'w':
                self._handle_cmd('forward')
            elif key == 's':
                self._handle_cmd('backward')
            elif key == 'a':
                self._handle_cmd('left')
            elif key == 'd':
                self._handle_cmd('right')
            elif key == 'x':
                self._handle_cmd('stop')
            elif key == 'q':
                self.get_logger().info("q 입력: 종료 요청")
                self._running = False
            else:
                pass

    # -------------------------
    # S-curve ramp
    # -------------------------
    @staticmethod
    def _smoothstep(u: float) -> float:
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return (3.0 * u * u) - (2.0 * u * u * u)

    def _start_ramp_to(self, left_target: int, right_target: int, duration: float = RAMP_TIME_SEC):
        self.left_target = int(left_target)
        self.right_target = int(right_target)

        self.left_start = int(self.left_pulse)
        self.right_start = int(self.right_pulse)

        self.ramp_t0 = time.monotonic()
        self.ramp_duration = max(0.01, float(duration))

        if self.ramp_timer is not None:
            self.ramp_timer.cancel()
            self.ramp_timer = None

        # ramp 구간: busy True
        self.publish_busy(True)

        self.ramp_timer = self.create_timer(RAMP_DT, self._ramp_step)

    def _ramp_step(self):
        t = time.monotonic() - self.ramp_t0
        u = t / self.ramp_duration
        s = self._smoothstep(u)

        self.left_pulse = int(round(self.left_start + (self.left_target - self.left_start) * s))
        self.right_pulse = int(round(self.right_start + (self.right_target - self.right_start) * s))

        self.set_pwm(LEFT_GPIO, self.left_pulse)
        self.set_pwm(RIGHT_GPIO, self.right_pulse)

        if u >= 1.0:
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

    # -------------------------
    # Motion commands
    # -------------------------
    def start_motion_with_auto_stop(self, motion_desc: str, left_pulse_us: int, right_pulse_us: int):
        if self.is_busy:
            self.get_logger().warn(f"BUSY: '{motion_desc}' 명령 무시 (동작 중)")
            return

        self.get_logger().info(
            f"명령: {motion_desc} (좌 {left_pulse_us}µs / 우 {right_pulse_us}µs) "
            f"-> ramp {RAMP_TIME_SEC}s"
        )

        self._start_ramp_to(left_pulse_us, right_pulse_us, duration=RAMP_TIME_SEC)

        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        self.stop_timer = self.create_timer(MOTION_DURATION_SEC, self.auto_stop_callback)

    def auto_stop_callback(self):
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        self.get_logger().info(f"{MOTION_DURATION_SEC}초 경과: 자동 정지(중립)로 ramp down")
        self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US, duration=RAMP_TIME_SEC)

    def _handle_cmd(self, cmd: str):
        cmd = cmd.strip().lower()

        if self.is_busy and cmd != 'stop':
            self.get_logger().warn(f"BUSY: '{cmd}' 명령 무시 (동작 중)")
            return

        if cmd == 'forward':
            self.start_motion_with_auto_stop("FORWARD", PULSE_FORWARD_US, PULSE_FORWARD_US)
        elif cmd == 'backward':
            self.start_motion_with_auto_stop("BACKWARD", PULSE_BACK_US, PULSE_BACK_US)
        elif cmd == 'left':
            self.start_motion_with_auto_stop("LEFT", PULSE_ROTATE_US_2, PULSE_ROTATE_US)
        elif cmd == 'right':
            self.start_motion_with_auto_stop("RIGHT", PULSE_ROTATE_US, PULSE_ROTATE_US_2)
        elif cmd == 'stop':
            self.get_logger().info("명령: STOP (중립으로 ramp down)")
            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None
            self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US, duration=RAMP_TIME_SEC)
        else:
            self.get_logger().warn(f"알 수 없는 명령: '{cmd}'")

    # -------------------------
    # Cleanup
    # -------------------------
    def cleanup(self):
        self.get_logger().info("노드 종료: 모터 정지 및 pigpio 정리.")

        self._running = False

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
            # 중립 출력 후 PWM off
            self.set_both_pwm(PULSE_STOP_US)
            time.sleep(0.3)
            self.pi.set_servo_pulsewidth(LEFT_GPIO, 0)
            self.pi.set_servo_pulsewidth(RIGHT_GPIO, 0)
        except Exception:
            pass
        finally:
            try:
                self.pi.stop()
            except Exception:
                pass

        self.publish_busy(False)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardMotorPwmNode()

    try:
        # 키 입력은 별도 쓰레드, ROS 타이머(ramp/auto-stop)는 spin으로 돌림
        while rclpy.ok() and node._running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()