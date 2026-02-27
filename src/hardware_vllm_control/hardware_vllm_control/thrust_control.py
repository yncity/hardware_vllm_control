#!/usr/bin/env python3
import time
import pigpio
import rclpy
import lgpio

from rclpy.node import Node
from std_msgs.msg import String, Bool

# =========================
# GPIO (BCM)
# =========================
LEFT_GPIO = 13
RIGHT_GPIO = 18

# =========================
# ESC pulse widths (microseconds)
# =========================
PULSE_FORWARD_US = 1350
PULSE_STOP_US    = 1500
PULSE_BACK_US    = 1650

PULSE_ROTATE_US  = 1400
PULSE_ROTATE_US_2 = 1600
# command hold (sec) then auto stop
MOTION_DURATION_SEC = 1.0

# =========================
# PWM ramp (S-curve)
# =========================
RAMP_TIME_SEC = 0.5
RAMP_HZ = 50
RAMP_DT = 1.0 / RAMP_HZ

# =========================
# LED (GPIO24)
# =========================
LED_GPIO = 24
BLINK_TOGGLE_DT = 0.25  # 0.25s toggle


class MotorPwmNode(Node):
    def __init__(self):
        super().__init__('motor_pwm_node')

        # pigpio connection
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon 연결 실패. 'sudo pigpiod' 실행 여부 확인")

        # ESC arming
        self.get_logger().info("초기 중립 신호(1500µs)를 양쪽 모터에 출력합니다 (arming).")
        self.set_both_pwm(PULSE_STOP_US)
        time.sleep(1.0)

        # ===== LED(GPIO24) =====
        self.led_handle = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.led_handle, LED_GPIO)
        self.led_state = 1
        lgpio.gpio_write(self.led_handle, LED_GPIO, 1)  # idle 기본 ON
        self.led_blink_timer = None

        # thrust_busy publish
        self.busy_pub = self.create_publisher(Bool, '/thrust_busy', 10)
        self.is_busy = False
        self.publish_busy(False)  # -> idle ON

        # cmd sub
        self.sub = self.create_subscription(String, '/cmd_motor', self.cmd_callback, 10)

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

        self.get_logger().info(
            "MotorPwmNode(pigpio) 시작.\n"
            "/cmd_motor: forward/backward/left/right/stop\n"
            f"램프: S-curve(3t^2-2t^3), {RAMP_TIME_SEC}s, {RAMP_HZ}Hz\n"
            f"자동정지: {MOTION_DURATION_SEC}s 후 중립\n"
            f"LED(GPIO{LED_GPIO}): busy 깜빡(0.25s), idle 항상 ON"
        )

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
    # Busy + LED policy
    # -------------------------
    def publish_busy(self, state: bool):
        self.is_busy = bool(state)
        self.busy_pub.publish(Bool(data=self.is_busy))

        # LED policy:
        # - busy: blink
        # - idle: steady ON
        if self.is_busy:
            self._start_led_blink()
        else:
            self._stop_led_blink(keep_on=True)

    def _start_led_blink(self):
        if self.led_blink_timer is not None:
            return
        self.led_blink_timer = self.create_timer(BLINK_TOGGLE_DT, self._toggle_led)

    def _stop_led_blink(self, keep_on: bool = True):
        if self.led_blink_timer is not None:
            self.led_blink_timer.cancel()
            self.led_blink_timer = None

        self.led_state = 1 if keep_on else 0
        try:
            lgpio.gpio_write(self.led_handle, LED_GPIO, self.led_state)
        except Exception:
            pass

    def _toggle_led(self):
        self.led_state = 0 if self.led_state else 1
        try:
            lgpio.gpio_write(self.led_handle, LED_GPIO, self.led_state)
        except Exception:
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

            # 목표가 중립이면 busy False (=> LED steady ON)
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
            f"-> S-curve 램프 {RAMP_TIME_SEC}s"
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

        self.get_logger().info(f"{MOTION_DURATION_SEC}초 경과: 자동 정지(중립)로 S-curve 램프 다운")
        self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US, duration=RAMP_TIME_SEC)

    def cmd_callback(self, msg: String):
        cmd = msg.data.strip().lower()

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
            self.get_logger().info("명령: STOP (중립으로 S-curve 램프 다운)")
            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None
            self._start_ramp_to(PULSE_STOP_US, PULSE_STOP_US, duration=RAMP_TIME_SEC)
        else:
            self.get_logger().warn(
                f"알 수 없는 명령: '{cmd}'. 사용 가능: forward / backward / left / right / stop"
            )

    # -------------------------
    # Cleanup
    # -------------------------
    def cleanup(self):
        self.get_logger().info("노드 종료: 모터 정지 및 pigpio/lgpio 정리.")

        try:
            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None
            if self.ramp_timer is not None:
                self.ramp_timer.cancel()
                self.ramp_timer = None
            if self.led_blink_timer is not None:
                self.led_blink_timer.cancel()
                self.led_blink_timer = None
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

        # LED는 종료 시 OFF로 정리(원하면 keep_on=True로 바꿔도 됨)
        try:
            lgpio.gpio_write(self.led_handle, LED_GPIO, 0)
        except Exception:
            pass
        try:
            lgpio.gpiochip_close(self.led_handle)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MotorPwmNode()
    try:
        rclpy.spin(node)
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
