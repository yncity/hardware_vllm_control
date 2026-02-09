#!/usr/bin/env python3
import os
import shlex
import subprocess

import rclpy
from rclpy.node import Node


class RpicamTimelapseNode(Node):
    def __init__(self):
        super().__init__('rpicam_timelapse_node')

        # ===== 설정 부분(필요시 숫자만 수정해서 사용) =====
        # 촬영 간격(초) - 예: 7.0초마다 1장
        self.interval_sec = 7.0

        # 저장 폴더
        self.output_dir = os.path.expanduser('~/saved_images')

        # 파일 이름 패턴: saved_image_000000.jpg 이런 식으로 저장
        self.filename_prefix = 'saved_image_'
        self.filename_ext = '.jpg'
        # ===============================================

        # 저장 폴더 생성 (없으면 만든다)
        os.makedirs(self.output_dir, exist_ok=True)

        # 현재까지 저장된 파일들 확인해서 다음 번호부터 시작
        self.counter = self.find_start_index()

        # 주기적으로 한 장씩 촬영하는 타이머
        self.timer = self.create_timer(self.interval_sec, self.capture_once)

        self.get_logger().info(
            f"[image_saver] 시작: {self.interval_sec}초마다 1장씩 캡처\n"
            f"저장 경로: {self.output_dir}, 시작 인덱스: {self.counter}"
        )

    def find_start_index(self) -> int:
        """
        이미 saved_images 폴더에 있는 파일들 중
        saved_image_XXXXXX.jpg 의 XXXXXX 중 가장 큰 값 + 1부터 시작.
        없으면 0부터 시작.
        """
        import re

        pattern = re.compile(
            rf"{self.filename_prefix}(\d+){self.filename_ext}"
        )
        max_idx = -1

        try:
            for f in os.listdir(self.output_dir):
                m = pattern.fullmatch(f)
                if m:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
        except FileNotFoundError:
            # 폴더가 없다면 이미 os.makedirs에서 만들기 때문에
            # 여기 들어올 일은 거의 없음
            pass

        return max_idx + 1

    def build_output_path(self) -> str:
        """
        현재 counter를 사용해 저장할 파일 전체 경로 생성
        예) saved_image_000000.jpg
        """
        filename = f"{self.filename_prefix}{self.counter:06d}{self.filename_ext}"
        return os.path.join(self.output_dir, filename)

    def capture_once(self):
        """
        타이머마다 한 번씩 호출되어, rpicam-jpeg로 사진 한 장 촬영
        """
        output_path = self.build_output_path()

        cmd = [
            'rpicam-jpeg',
            '-n',          # 프리뷰 창 끄기 (창 안 띄우고 찍기)
            '-t', '1000',  # 1초 동안 노출 후 한 장 찍기
            '-o', output_path,
        ]

        self.get_logger().info(
            f"[image_saver] 캡처 명령 실행: {' '.join(shlex.quote(c) for c in cmd)}"
        )

        try:
            # 블로킹으로 한 번만 실행
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

            if result.returncode == 0:
                self.get_logger().info(
                    f"[image_saver] 캡처 성공 → {output_path}"
                )
                self.counter += 1
            else:
                self.get_logger().error(
                    "[image_saver] 캡처 실패\n"
                    f"  return code: {result.returncode}\n"
                    f"  stdout: {result.stdout}\n"
                    f"  stderr: {result.stderr}"
                )

        except FileNotFoundError:
            self.get_logger().error(
                "[image_saver] rpicam-jpeg 명령을 찾을 수 없습니다. "
                "PATH 및 설치 상태를 확인하세요."
            )
        except Exception as e:
            self.get_logger().error(f"[image_saver] 예외 발생: {e}")


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