#!/usr/bin/env python3
import os
import shlex
import subprocess

import rclpy
from rclpy.node import Node

import lgpio


class RpicamTimelapseNode(Node):
    def __init__(self):
        super().__init__('rpicam_timelapse_node')

        # ===== GPIO 설정 =====
        self.led_pins = [23, 24, 27, 22, 21]

        # GPIO 초기화
        self.gpio_handle = lgpio.gpiochip_open(0)

        for pin in self.led_pins:
            lgpio.gpio_claim_output(self.gpio_handle, pin)
            lgpio.gpio_write(self.gpio_handle, pin, 1)  # 항상 ON

        self.get_logger().info("[GPIO] LED 항상 점등 상태로 설정 완료")

        # ===== 카메라 설정 =====
        self.interval_sec = 7.0
        self.output_dir = os.path.expanduser('~/saved_images')
        self.filename_prefix = 'saved_image_'
        self.filename_ext = '.jpg'

        os.makedirs(self.output_dir, exist_ok=True)

        self.counter = self.find_start_index()
        self.timer = self.create_timer(self.interval_sec, self.capture_once)

        self.get_logger().info(
            f"[image_saver] 시작: {self.interval_sec}초마다 1장씩 캡처\n"
            f"저장 경로: {self.output_dir}, 시작 인덱스: {self.counter}"
        )

    def find_start_index(self) -> int:
        import re
        pattern = re.compile(
            rf"{self.filename_prefix}(\d+){self.filename_ext}"
        )
        max_idx = -1

        for f in os.listdir(self.output_dir):
            m = pattern.fullmatch(f)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx

        return max_idx + 1

    def build_output_path(self) -> str:
        filename = f"{self.filename_prefix}{self.counter:06d}{self.filename_ext}"
        return os.path.join(self.output_dir, filename)

    def capture_once(self):
        output_path = self.build_output_path()

        cmd = [
            'rpicam-jpeg',
            '-n',
            '-t', '1000',
            '--width', '1920',
            '--height', '1080',
            '-q', '85',
            '-o', output_path,
        ]

        self.get_logger().info(
            f"[image_saver] 캡처 실행: {' '.join(shlex.quote(c) for c in cmd)}"
        )

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if result.returncode == 0:
                self.get_logger().info(f"[image_saver] 성공 → {output_path}")
                self.counter += 1
            else:
                self.get_logger().error(
                    f"[image_saver] 실패 code={result.returncode}"
                )

        except Exception as e:
            self.get_logger().error(f"[image_saver] 예외: {e}")

    def destroy_node(self):
        # 종료 시 LED OFF
        self.get_logger().info("[GPIO] LED OFF 및 GPIO 정리")

        for pin in self.led_pins:
            lgpio.gpio_write(self.gpio_handle, pin, 0)

        lgpio.gpiochip_close(self.gpio_handle)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RpicamTimelapseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
